package com.example.myapplication

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.util.ArrayList

class AdvancedBurstProcessor {

    private val TAG = "AdvancedBurstProcessor"

    init {
        try {
            System.loadLibrary("opencv_java4")
            Log.i(TAG, "[DIAGNOSTIC] OpenCV loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "[DIAGNOSTIC] Failed to load OpenCV", e)
        }
    }

    private val capturedFrames = mutableListOf<FrameData>()

    data class FrameData(val mat: Mat, val sharpness: Double)

    fun addFrame(bitmap: Bitmap) {
        val color = Mat()
        try {
            Utils.bitmapToMat(bitmap, color)
            Imgproc.cvtColor(color, color, Imgproc.COLOR_RGBA2BGR)
        } finally {
            bitmap.recycle() // Liberar memoria del Bitmap INMEDIATAMENTE para evitar OOM
        }

        if (color.empty()) {
            color.release()
            return
        }

        val gray = Mat()
        val laplacian = Mat()
        val mean = MatOfDouble()
        val stddev = MatOfDouble()

        try {
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
            Core.meanStdDev(laplacian, mean, stddev)
            
            // 1. Métrica de nitidez normalizada
            val lapVar = Math.pow(stddev.get(0, 0)[0], 2.0)
            val grayMean = Core.mean(gray).`val`[0]
            val sharpness = if (grayMean > 1.0) lapVar / grayMean else lapVar
            
            capturedFrames.add(FrameData(color, sharpness))
            Log.d(TAG, "Frame added. Sharpness: $sharpness. Total frames: ${capturedFrames.size}")
        } finally {
            gray.release()
            laplacian.release()
            mean.release()
            stddev.release()
        }
    }

    fun getFrameCount(): Int = capturedFrames.size

    fun process(): ByteArray? {
        if (capturedFrames.isEmpty()) return null
        Log.i(TAG, "Processing ${capturedFrames.size} frames with MFSR & Lucky Imaging")

        // 1. Lucky Imaging: Seleccionar los 15 mejores frames
        capturedFrames.sortByDescending { it.sharpness }
        val topFrames = capturedFrames.take(15).map { it.mat }
        
        // Liberar memoria de los descartados
        capturedFrames.drop(15).forEach { it.mat.release() }
        capturedFrames.clear()

        if (topFrames.isEmpty()) return null

        val refFrame = topFrames[0]
        val refGray = Mat()
        val refGraySmall = Mat()
        
        Imgproc.cvtColor(refFrame, refGray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.resize(refGray, refGraySmall, Size(), 0.25, 0.25, Imgproc.INTER_AREA)

        val alignedMats = mutableListOf<Mat>()
        alignedMats.add(refFrame.clone())

        val w = refFrame.cols()
        val h = refFrame.rows()
        val mapXBase = Mat(h, w, CvType.CV_32FC1)
        val mapYBase = Mat(h, w, CvType.CV_32FC1)
        
        val mapXarr = FloatArray(w * h)
        val mapYarr = FloatArray(w * h)
        var idx = 0
        for (r in 0 until h) {
            for (c in 0 until w) {
                mapXarr[idx] = c.toFloat()
                mapYarr[idx] = r.toFloat()
                idx++
            }
        }
        mapXBase.put(0, 0, mapXarr)
        mapYBase.put(0, 0, mapYarr)

        // 2. Alineación Densa (Optical Flow Farneback)
        for (i in 1 until topFrames.size) {
            val altFrame = topFrames[i]
            val altGray = Mat()
            val altGraySmall = Mat()
            val flowSmall = Mat()
            val flowLarge = Mat()
            val mapX = Mat()
            val mapY = Mat()
            val warpedAlt = Mat()
            val flowChannels = ArrayList<Mat>(2)

            try {
                Imgproc.cvtColor(altFrame, altGray, Imgproc.COLOR_BGR2GRAY)
                Imgproc.resize(altGray, altGraySmall, Size(), 0.25, 0.25, Imgproc.INTER_AREA)

                Video.calcOpticalFlowFarneback(
                    refGraySmall, altGraySmall, flowSmall,
                    0.5, 5, 15, 3, 7, 1.5, 0
                )

                Imgproc.resize(flowSmall, flowLarge, refFrame.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
                Core.multiply(flowLarge, Scalar(4.0, 4.0), flowLarge)

                Core.split(flowLarge, flowChannels)
                Core.add(mapXBase, flowChannels[0], mapX)
                Core.add(mapYBase, flowChannels[1], mapY)

                Imgproc.remap(altFrame, warpedAlt, mapX, mapY, Imgproc.INTER_LANCZOS4)
                alignedMats.add(warpedAlt.clone())
                Log.d(TAG, "Frame $i aligned via Optical Flow.")
            } finally {
                altGray.release()
                altGraySmall.release()
                flowSmall.release()
                flowLarge.release()
                warpedAlt.release()
                mapX.release()
                mapY.release()
                flowChannels.forEach { it.release() }
            }
        }

        refGray.release()
        refGraySmall.release()
        mapXBase.release()
        mapYBase.release()

        // 3. Fusión Robusta con Máscara Anti-Ghosting
        val accumulator = Mat.zeros(refFrame.size(), CvType.CV_32FC3)
        val weightSum = Mat.zeros(refFrame.size(), CvType.CV_32FC1)
        val refFloat = Mat()
        
        try {
            refFrame.convertTo(refFloat, CvType.CV_32FC3)

            for (aligned in alignedMats) {
                val alignedFloat = Mat()
                val diff = Mat()
                val diffGray = Mat()
                val weight = Mat()
                val weight3C = Mat()
                val weightedFrame = Mat()

                try {
                    aligned.convertTo(alignedFloat, CvType.CV_32FC3)
                    Core.absdiff(refFloat, alignedFloat, diff)
                    Imgproc.cvtColor(diff, diffGray, Imgproc.COLOR_BGR2GRAY)
                    Imgproc.GaussianBlur(diffGray, diffGray, Size(5.0, 5.0), 1.5)

                    Core.add(diffGray, Scalar(10.0), weight)
                    Core.divide(1.0, weight, weight) // weight = 1 / (diff + 10)

                    Core.merge(listOf(weight, weight, weight), weight3C)
                    Core.multiply(alignedFloat, weight3C, weightedFrame)

                    Core.add(accumulator, weightedFrame, accumulator)
                    Core.add(weightSum, weight, weightSum)
                } finally {
                    alignedFloat.release()
                    diff.release()
                    diffGray.release()
                    weight.release()
                    weight3C.release()
                    weightedFrame.release()
                }
            }
        } finally {
            refFloat.release()
        }

        val weightSum3C = Mat()
        val fused = Mat()
        
        try {
            Core.merge(listOf(weightSum, weightSum, weightSum), weightSum3C)
            Core.divide(accumulator, weightSum3C, accumulator)
            accumulator.convertTo(fused, CvType.CV_8UC3)
        } finally {
            accumulator.release()
            weightSum.release()
            weightSum3C.release()
        }

        // 2. Recorte de bordes tras la fusión
        var cropX = 8
        var cropY = 8
        var cropW = fused.cols() - 16
        var cropH = fused.rows() - 16
        if (cropW <= 0 || cropH <= 0) {
            cropX = 0
            cropY = 0
            cropW = fused.cols()
            cropH = fused.rows()
        }
        val rect = Rect(cropX, cropY, cropW, cropH)
        val fusedCropped = Mat(fused, rect)

        // 4. Realce de Contraste Adaptativo (CLAHE), Unsharp Masking y Codificación
        val lab = Mat()
        val labChannels = ArrayList<Mat>(3)
        val claheL = Mat()
        val enhancedBGR = Mat()
        val blurred = Mat()
        val sharpened = Mat()
        val finalMat = Mat()
        val outBuf = MatOfByte()
        val params = MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92)

        try {
            Imgproc.cvtColor(fusedCropped, lab, Imgproc.COLOR_BGR2Lab)
            Core.split(lab, labChannels)

            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(labChannels[0], claheL)
            clahe.clear()

            labChannels[0].release()
            labChannels[0] = claheL

            Core.merge(labChannels, lab)
            Imgproc.cvtColor(lab, enhancedBGR, Imgproc.COLOR_Lab2BGR)

            // 3. Unsharp masking después de CLAHE
            Imgproc.GaussianBlur(enhancedBGR, blurred, Size(0.0, 0.0), 1.5)
            Core.addWeighted(enhancedBGR, 1.5, blurred, -0.5, 0.0, sharpened)

            val maxSide = 2048
            val width = sharpened.cols()
            val height = sharpened.rows()

            if (width > maxSide || height > maxSide) {
                val scale = if (width > height) maxSide.toDouble() / width else maxSide.toDouble() / height
                Imgproc.resize(sharpened, finalMat, Size(), scale, scale, Imgproc.INTER_AREA)
            } else {
                sharpened.copyTo(finalMat)
            }

            Imgcodecs.imencode(".jpg", finalMat, outBuf, params)

            Log.i(TAG, "MFSR Processing Complete. Output size: ${outBuf.toArray().size} bytes.")
            return outBuf.toArray()
        } finally {
            fused.release()
            fusedCropped.release()
            lab.release()
            labChannels.forEach { it.release() }
            enhancedBGR.release()
            blurred.release()
            sharpened.release()
            finalMat.release()
            outBuf.release()
            params.release()

            topFrames.forEach { it.release() }
            alignedMats.forEach { it.release() }
        }
    }

    fun release() {
        capturedFrames.forEach { it.mat.release() }
        capturedFrames.clear()
    }
}