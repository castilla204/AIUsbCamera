import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

fun loadSecretProps(): Properties {
    val props = Properties()
    // Defaults (empty) committed; real values from secrets.properties / local.properties / env
    listOf(
        rootProject.file("secrets.defaults.properties"),
        rootProject.file("secrets.properties"),
        rootProject.file("local.properties"),
    ).forEach { file ->
        if (file.exists()) {
            file.inputStream().use { props.load(it) }
        }
    }
    // CI: GitHub Actions can inject via env without writing files
    listOf(
        "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "API_KEY_RELAY", "RELAY_URL_RENDER", "RELAY_URL_RAILWAY",
    ).forEach { key ->
        System.getenv(key)?.takeIf { it.isNotBlank() }?.let { props[key] = it }
    }
    return props
}

fun Properties.secret(name: String): String =
    getProperty(name)?.trim()?.replace("\\", "\\\\")?.replace("\"", "\\\"") ?: ""

val secretProps = loadSecretProps()

android {
    namespace = "com.example.myapplication"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.myapplication"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "GEMINI_API_KEY", "\"${secretProps.secret("GEMINI_API_KEY")}\"")
        buildConfigField("String", "ANTHROPIC_API_KEY", "\"${secretProps.secret("ANTHROPIC_API_KEY")}\"")
        buildConfigField("String", "OPENAI_API_KEY", "\"${secretProps.secret("OPENAI_API_KEY")}\"")
        buildConfigField("String", "API_KEY_RELAY", "\"${secretProps.secret("API_KEY_RELAY")}\"")
        buildConfigField("String", "RELAY_URL_RENDER", "\"${secretProps.secret("RELAY_URL_RENDER")}\"")
        buildConfigField("String", "RELAY_URL_RAILWAY", "\"${secretProps.secret("RELAY_URL_RAILWAY")}\"")
    }


    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        buildConfig = true
        viewBinding = true
        aidl = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    // API de Xposed/LSPosed — SOLO compilación (compileOnly): LSPosed inyecta la
    // implementación real en runtime. Jar ORIGINAL del API 82 (paquete de.robv.*,
    // Java puro, ~25KB) en app/libs/. Razón: jcenter/api.xposed.info están muertos
    // y el mirror de Maven Central traía metadata de Kotlin incompatible.
    compileOnly(files("libs/xposed-api-82.jar"))

    // Lifecycle and Coroutines for ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.0")
    implementation("androidx.activity:activity-ktx:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.0")
    
    // Usando librería local clonada de serenegiant/UVCCamera
    // Exclude com.android.support to avoid conflicts with AndroidX
    implementation("com.serenegiant:common:${rootProject.extra["commonLibVersion"]}") {
        exclude(group = "com.android.support")
    }
    // Exclude com.android.support from libuvccamera's transitive dependencies
    implementation(project(":libuvccamera")) {
        exclude(group = "com.android.support")
    }
    
    // HTTP & JSON
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.moshi:moshi:1.15.1")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.1")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // OpenCV
    implementation("org.opencv:opencv:4.13.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
