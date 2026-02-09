package com.example.myapplication

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.util.Log
import android.view.Surface

/**
 * A helper class to create an off-screen EGL context and SurfaceTexture.
 * This tricks native camera libraries (like UVCCamera) into thinking they are rendering to a real
 * screen, forcing their firmware to turn on auto-exposure and auto-white-balance.
 */
class OffscreenSurface(private val width: Int, private val height: Int) {

    private val TAG = "OffscreenSurface"

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    
    private var textureId = -1
    private var surfaceTexture: SurfaceTexture? = null
    var surface: Surface? = null
        private set

    init {
        setupEgl()
    }

    private fun setupEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "unable to get EGL14 display")
            return
        }

        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            Log.e(TAG, "unable to initialize EGL14")
            return
        }

        val attribList = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, configs.size, numConfigs, 0)
        
        val config = configs[0]
        if (config == null) {
            Log.e(TAG, "Unable to find a suitable EGLConfig")
            return
        }

        val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

        // IMPORTANTE: Un PbufferSurface real en lugar de 1x1. UVCCamera puede requerirlo.
        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, width, 
            EGL14.EGL_HEIGHT, height, 
            EGL14.EGL_NONE
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, config, surfaceAttribs, 0)

        makeCurrent()

        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR.toFloat())
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR.toFloat())
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        surfaceTexture = SurfaceTexture(textureId).apply {
            setDefaultBufferSize(width, height)
            setOnFrameAvailableListener { st ->
                try {
                    makeCurrent()
                    st.updateTexImage()
                } catch (e: Exception) {
                    Log.e(TAG, "Error updating texture image in background", e)
                }
            }
        }
        surface = Surface(surfaceTexture)
        Log.i(TAG, "Offscreen EGL context and surface initialized successfully.")
    }

    fun makeCurrent() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY && eglSurface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
        }
    }

    fun release() {
        Log.i(TAG, "Releasing OffscreenSurface.")
        
        surfaceTexture?.setOnFrameAvailableListener(null)
        
        surface?.release()
        surface = null
        
        surfaceTexture?.release()
        surfaceTexture = null

        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (textureId != -1) {
                GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
                textureId = -1
            }
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglReleaseThread()
            EGL14.eglTerminate(eglDisplay)
        }

        eglDisplay = EGL14.EGL_NO_DISPLAY
        eglContext = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE
    }
}

object GLES11Ext {
    const val GL_TEXTURE_EXTERNAL_OES = 0x8D65
}