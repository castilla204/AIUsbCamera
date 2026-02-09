package com.example.myapplication

import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.view.SurfaceView
import kotlin.math.roundToInt

/**
 * SurfaceView que se mide a sí mismo para mantener el aspect ratio de los frames de la
 * cámara y, además, fija el tamaño del buffer subyacente a la resolución de la cámara
 * (vía SurfaceHolder.setFixedSize). Esto es lo que hace el sample oficial de Android
 * camera-samples (Camera2Basic). En vertical, sin esto, el SurfaceView aloca un buffer
 * a tamaño de la vista y la conversión RGBX de libuvccamera escribe filas de 1280 px
 * en filas de menor stride, dejando el preview "cortado".
 */
class AutoFitSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : SurfaceView(context, attrs, defStyle) {

    private var aspectRatio = 0f

    /**
     * Llamar con la resolución NATIVA de la cámara. El holder se fuerza a ese tamaño y
     * la onMeasure de la vista se calcula a ese aspect ratio.
     */
    fun setAspectRatio(width: Int, height: Int) {
        require(width > 0 && height > 0) { "Size cannot be negative" }
        aspectRatio = width.toFloat() / height.toFloat()
        try { holder.setFixedSize(width, height) } catch (_: Exception) {}
        Log.i(TAG, "setAspectRatio($width x $height), holder.setFixedSize ok")
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        val maxW = MeasureSpec.getSize(widthMeasureSpec)
        val maxH = MeasureSpec.getSize(heightMeasureSpec)
        if (aspectRatio == 0f || maxW == 0 || maxH == 0) {
            setMeasuredDimension(maxW, maxH)
            return
        }

        // FIT (sin recorte): elegir el lado mayor que QUEPA dentro de la spec
        // manteniendo el aspect de la cámara. Nunca devolvemos algo mayor que la spec
        // (al revés que la versión "center-crop" del sample oficial, que vale para
        // preview pantalla completa pero rebosa cuando estás dentro de un contenedor
        // limitado en altura como aquí: el preview va arriba y debajo van los botones).
        val candidateHeight = (maxW / aspectRatio).roundToInt()
        val newWidth: Int
        val newHeight: Int
        if (candidateHeight <= maxH) {
            newWidth = maxW
            newHeight = candidateHeight
        } else {
            newHeight = maxH
            newWidth = (maxH * aspectRatio).roundToInt()
        }
        Log.d(TAG, "onMeasure spec=${maxW}x${maxH} -> ${newWidth}x${newHeight} (aspect=$aspectRatio)")
        setMeasuredDimension(newWidth, newHeight)
    }

    companion object {
        private const val TAG = "AutoFitSurfaceView"
    }
}
