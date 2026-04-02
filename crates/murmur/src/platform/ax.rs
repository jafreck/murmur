//! macOS Accessibility and Core Foundation FFI helpers.
//!
//! **All raw `unsafe` AX/CF FFI is quarantined in this module.**  Other
//! modules should use the safe wrappers below instead of writing their own
//! `extern "C"` blocks or `unsafe` calls.

use std::ffi::c_void;
use std::ptr;

// ── Type aliases ───────────────────────────────────────────────────────

pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFIndex = isize;
pub type AXUIElementRef = CFTypeRef;
pub type AXValueRef = CFTypeRef;
pub type AXError = i32;

pub const K_AX_ERROR_SUCCESS: AXError = 0;
pub const K_AX_VALUE_TYPE_CF_RANGE: u32 = 4;

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CFRange {
    pub location: CFIndex,
    pub length: CFIndex,
}

// ── FFI declarations (private) ─────────────────────────────────────────

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
    fn AXValueCreate(value_type: u32, value_ptr: *const c_void) -> AXValueRef;
    fn AXValueGetValue(value: AXValueRef, value_type: u32, value_ptr: *mut c_void) -> bool;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFStringGetLength(the_string: CFStringRef) -> CFIndex;
    fn CFStringGetCString(
        the_string: CFStringRef,
        buffer: *mut u8,
        buffer_size: CFIndex,
        encoding: u32,
    ) -> bool;
    fn CFStringCreateWithCString(alloc: CFTypeRef, c_str: *const u8, encoding: u32) -> CFStringRef;
}

// ── RAII guard ─────────────────────────────────────────────────────────

/// RAII guard that calls `CFRelease` on drop.
///
/// Wraps an owned (+1 retained) Core Foundation reference. The inner
/// pointer is public so callers can pass it to other AX/CF functions.
pub struct CfGuard(pub CFTypeRef);

impl CfGuard {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

impl Drop for CfGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 is a +1 retained Core Foundation reference
            // obtained from a CF "Create" or "Copy" function. Releasing it
            // exactly once here balances the retain count.
            unsafe { CFRelease(self.0) };
        }
    }
}

// ── Safe wrappers ──────────────────────────────────────────────────────

/// Create a `CFStringRef` from a NUL-terminated byte literal.
///
/// Returns a [`CfGuard`] that releases the string on drop.
pub fn cfstring_create(s: &[u8]) -> CfGuard {
    // SAFETY: `s` is a NUL-terminated byte slice (caller invariant).
    // CFStringCreateWithCString reads up to the NUL and returns a +1
    // retained CFString.
    let ptr =
        unsafe { CFStringCreateWithCString(ptr::null(), s.as_ptr(), K_CF_STRING_ENCODING_UTF8) };
    CfGuard(ptr)
}

/// Convert a `CFStringRef` to a Rust `String`.
///
/// Returns `None` if the reference is null or conversion fails.
pub fn cfstring_to_string(cf_str: CFStringRef) -> Option<String> {
    if cf_str.is_null() {
        return None;
    }
    // SAFETY: cf_str is a valid CFStringRef (caller-guaranteed).
    // CFStringGetLength and CFStringGetCString only read from the string.
    unsafe {
        let len = CFStringGetLength(cf_str);
        // UTF-8 can use up to 4 bytes per code point, plus a NUL terminator.
        let max_size = len * 4 + 1;
        let mut buffer = vec![0u8; max_size as usize];
        if CFStringGetCString(
            cf_str,
            buffer.as_mut_ptr(),
            max_size,
            K_CF_STRING_ENCODING_UTF8,
        ) {
            let nul_pos = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
            Some(String::from_utf8_lossy(&buffer[..nul_pos]).into_owned())
        } else {
            None
        }
    }
}

/// Copy an accessibility attribute value from `element`.
///
/// Returns `Some(CfGuard)` wrapping the owned reference, or `None` if the
/// attribute does not exist or the call fails.
pub fn ax_copy_attr(element: AXUIElementRef, attr: CFStringRef) -> Option<CfGuard> {
    // SAFETY: element and attr must be valid CF references (caller
    // invariant). On success the function writes a +1 retained reference
    // into `value`; we wrap it in CfGuard for automatic release.
    unsafe {
        let mut value: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(element, attr, &mut value);
        if err == K_AX_ERROR_SUCCESS && !value.is_null() {
            Some(CfGuard(value))
        } else {
            None
        }
    }
}

/// Set an accessibility attribute on `element`.
///
/// Returns the AX error code (`K_AX_ERROR_SUCCESS` = 0 on success).
pub fn ax_set_attr(element: AXUIElementRef, attr: CFStringRef, value: CFTypeRef) -> AXError {
    // SAFETY: element, attr, and value must be valid CF references
    // (caller invariant). The function does not take ownership.
    unsafe { AXUIElementSetAttributeValue(element, attr, value) }
}

/// Create the system-wide accessibility element.
pub fn ax_system_wide() -> CfGuard {
    // SAFETY: No preconditions. Returns a +1 retained AXUIElementRef.
    CfGuard(unsafe { AXUIElementCreateSystemWide() })
}

/// Create an accessibility element for an application by PID.
pub fn ax_application(pid: i32) -> CfGuard {
    // SAFETY: The PID should refer to a running process; a stale PID
    // simply causes subsequent attribute queries to fail gracefully.
    // Returns a +1 retained AXUIElementRef.
    CfGuard(unsafe { AXUIElementCreateApplication(pid) })
}

/// Extract a [`CFRange`] from an `AXValueRef`.
pub fn ax_value_to_range(value: AXValueRef) -> Option<CFRange> {
    // SAFETY: value must be an AXValueRef (caller invariant).
    // AXValueGetValue writes into `range` and returns false if the stored
    // type does not match K_AX_VALUE_TYPE_CF_RANGE.
    unsafe {
        let mut range = CFRange {
            location: 0,
            length: 0,
        };
        if AXValueGetValue(
            value,
            K_AX_VALUE_TYPE_CF_RANGE,
            &mut range as *mut CFRange as *mut c_void,
        ) {
            Some(range)
        } else {
            None
        }
    }
}

/// Create an `AXValueRef` containing a [`CFRange`].
pub fn ax_value_create_range(range: &CFRange) -> CfGuard {
    // SAFETY: AXValueCreate reads sizeof(CFRange) bytes from the pointer
    // and returns a +1 retained AXValueRef.
    CfGuard(unsafe {
        AXValueCreate(
            K_AX_VALUE_TYPE_CF_RANGE,
            range as *const CFRange as *const c_void,
        )
    })
}
