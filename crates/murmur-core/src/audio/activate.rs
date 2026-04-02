//! Platform-specific audio input device activation.
//!
//! On macOS, Bluetooth devices (like AirPods) connect in A2DP mode which
//! provides high-quality audio *output* but no microphone input.  The mic
//! requires a switch to the HFP/SCO profile, which macOS normally triggers
//! when an app selects the device as the system input.
//!
//! Higher-level Apple frameworks (`AVAudioSession`, `AVCaptureSession`)
//! handle this automatically, but the low-level AudioUnit HAL that `cpal`
//! uses does not always trigger the switch.
//!
//! This module provides a best-effort activation hook that re-sets the
//! default input device via CoreAudio, nudging macOS into establishing the
//! SCO link.  It is a no-op on non-macOS platforms.

/// Attempt to activate the system default input device for audio capture.
///
/// On macOS this re-sets the default input device via CoreAudio to nudge
/// Bluetooth devices into HFP mode.  On other platforms this is a no-op.
///
/// This is a best-effort operation — failures are logged at debug level
/// and do not propagate errors.
pub fn prepare_default_input() {
    #[cfg(target_os = "macos")]
    macos::activate_default_input();
}

// ── macOS CoreAudio implementation ──────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use std::os::raw::c_void;

    // CoreAudio HAL types
    type AudioObjectID = u32;
    type AudioDeviceID = u32;
    type OSStatus = i32;

    const K_AUDIO_OBJECT_SYSTEM_OBJECT: AudioObjectID = 1;

    // Property selectors (FourCC encoded)
    const K_AUDIO_HARDWARE_PROPERTY_DEFAULT_INPUT_DEVICE: u32 = u32::from_be_bytes(*b"dIn ");
    const K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL: u32 = u32::from_be_bytes(*b"glob");
    const K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN: u32 = 0;

    #[repr(C)]
    struct AudioObjectPropertyAddress {
        selector: u32,
        scope: u32,
        element: u32,
    }

    #[link(name = "CoreAudio", kind = "framework")]
    extern "C" {
        fn AudioObjectGetPropertyData(
            object_id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifier_data_size: u32,
            qualifier_data: *const c_void,
            data_size: *mut u32,
            data: *mut c_void,
        ) -> OSStatus;

        fn AudioObjectSetPropertyData(
            object_id: AudioObjectID,
            address: *const AudioObjectPropertyAddress,
            qualifier_data_size: u32,
            qualifier_data: *const c_void,
            data_size: u32,
            data: *const c_void,
        ) -> OSStatus;
    }

    /// Re-set the current default input device via CoreAudio.
    ///
    /// Writing the same device ID back to `kAudioHardwarePropertyDefaultInputDevice`
    /// can trigger macOS to establish the Bluetooth SCO/HFP link if it hasn't
    /// already.  This mirrors what System Settings does when the user selects
    /// a Bluetooth input device.
    pub(super) fn activate_default_input() {
        let addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_HARDWARE_PROPERTY_DEFAULT_INPUT_DEVICE,
            scope: K_AUDIO_OBJECT_PROPERTY_SCOPE_GLOBAL,
            element: K_AUDIO_OBJECT_PROPERTY_ELEMENT_MAIN,
        };

        let mut device_id: AudioDeviceID = 0;
        let mut size = std::mem::size_of::<AudioDeviceID>() as u32;

        // SAFETY: We pass the system audio object with the correct property
        // address for the default input device.  `device_id` is a valid
        // stack-local AudioDeviceID and `size` matches its byte width.
        // No qualifier data is needed for this property.
        let status = unsafe {
            AudioObjectGetPropertyData(
                K_AUDIO_OBJECT_SYSTEM_OBJECT,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut device_id as *mut _ as *mut c_void,
            )
        };

        if status != 0 {
            log::debug!("CoreAudio: failed to get default input device (status={status})");
            return;
        }

        // SAFETY: Same property address as the read above.  `device_id`
        // contains the value just read, and its size matches the property
        // type.  Writing the same device ID back is the intended nudge to
        // trigger Bluetooth HFP profile activation.
        let status = unsafe {
            AudioObjectSetPropertyData(
                K_AUDIO_OBJECT_SYSTEM_OBJECT,
                &addr,
                0,
                std::ptr::null(),
                std::mem::size_of::<AudioDeviceID>() as u32,
                &device_id as *const _ as *const c_void,
            )
        };

        if status != 0 {
            log::debug!("CoreAudio: failed to re-set default input device (status={status})");
        } else {
            log::info!("CoreAudio: activated default input device (id={device_id})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_default_input_does_not_panic() {
        // Should be a no-op on non-macOS, best-effort on macOS.
        prepare_default_input();
    }
}
