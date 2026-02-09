package com.example.myapplication

import java.nio.ByteBuffer

/**
 * Interface for receiving frames from the UVC camera.
 * Assumed to be implemented by the native/Java UVCPreview wrapper.
 */
interface IFrameCallback {
    fun onFrame(frameBuffer: ByteBuffer)
}
