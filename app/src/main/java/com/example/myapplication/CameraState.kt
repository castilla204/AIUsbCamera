package com.example.myapplication

sealed class CameraState {
    object Idle : CameraState() // No camera connected or initialized
    object Initializing : CameraState() // Camera being set up
    object ReadyForCapture : CameraState() // Camera connected, but not streaming
    object CapturingSinglePhoto : CameraState() // Currently taking one photo
    data class CapturingBurst(val capturedCount: Int, val totalCount: Int) : CameraState() // Taking multiple photos
    data class ProcessingImage(val message: String) : CameraState() // Performing heavy image processing
    data class Error(val message: String) : CameraState() // An error occurred
    object Stopping : CameraState() // Camera is in the process of stopping
}
