package com.example.myapplication

import android.graphics.Bitmap
import android.util.Base64
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.delay
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

class GeminiRestApiClient {

    // Tomar la key desde prefs/UI; no hardcodear secrets en el repo.
    private val apiKey: String
        get() = BolsilloIaClient.GEMINI_KEY

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val moshi = Moshi.Builder()
        .add(KotlinJsonAdapterFactory())
        .build()

    private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()

    suspend fun analyzeImage(bitmap: Bitmap, prompt: String, maxRetries: Int = 3): String? {
        val base64Image = encodeImageToBase64(bitmap)
        val url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

        val jsonBody = """
            {
              "contents": [
                {
                  "parts": [
                    { "text": "$prompt" },
                    {
                      "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": "$base64Image"
                      }
                    }
                  ]
                }
              ]
            }
        """.trimIndent()

        var currentRetry = 0
        var lastException: Exception? = null

        while (currentRetry <= maxRetries) {
            try {
                val request = Request.Builder()
                    .url(url)
                    .addHeader("X-goog-api-key", apiKey)
                    .post(jsonBody.toRequestBody(JSON_MEDIA_TYPE))
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string()
                    if (response.isSuccessful) {
                        return parseGeminiResponse(responseBody ?: return null)
                    } else if (response.code == 429 || response.code >= 500) {
                        // Retry on rate limit or server errors
                        val waitTime = Math.pow(2.0, currentRetry.toDouble()).toLong() * 1000
                        delay(waitTime)
                        currentRetry++
                    } else {
                        return "Error ${response.code}: ${responseBody ?: response.message}"
                    }
                }
            } catch (e: Exception) {
                lastException = e
                val waitTime = Math.pow(2.0, currentRetry.toDouble()).toLong() * 1000
                delay(waitTime)
                currentRetry++
            }
        }

        return "Failed after $maxRetries retries. Last exception: ${lastException?.message}"
    }

    private fun encodeImageToBase64(bitmap: Bitmap): String {
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 480, true)
        val outputStream = ByteArrayOutputStream()
        // JPEG 70% as requested
        resized.compress(Bitmap.CompressFormat.JPEG, 70, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    private fun parseGeminiResponse(json: String): String? {
        return try {
            val adapter = moshi.adapter(GeminiResponse::class.java)
            val response = adapter.fromJson(json)
            response?.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.text
        } catch (e: Exception) {
            e.printStackTrace()
            "Parse Error: ${e.message}"
        }
    }
}

data class GeminiResponse(val candidates: List<Candidate>?)
data class Candidate(val content: Content?)
data class Content(val parts: List<Part>?)
data class Part(val text: String?)