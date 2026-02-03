// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}

// Variables extra
extra.apply {

    set("androidXVersion", "1.3.1")
    set("versionCompiler", 35) 
    set("versionTarget", 35)
    set("minSdkVersion", 26)
    set("versionCode", 1)
    set("versionNameString", "3.3.5")
    set("javaSourceCompatibility", JavaVersion.VERSION_11)
    set("javaTargetCompatibility", JavaVersion.VERSION_11)
    set("supportLibVersion", "27.1.1")
    set("commonLibVersion", "2.12.4")
    set("versionBuildTool", "35.0.0")
    set("kotlinCoreVersion", "1.3.2")
    set("kotlinCoroutines", "1.3.9")
    set("materialVersion", "1.12.0")
    set("constraintlayoutVersion", "2.1.4")
    set("lifecycle_version", "2.2.0")
    set("quick_version", "2.9.50")
    set("dialog_version", "3.2.1")
    set("bugly_version", "3.4.4")
    set("bugly_native_version", "3.9.0")
    set("xlogVersion", "1.11.0")
}

tasks.register<Exec>("cloneNuuneoiUvc") {
    group = "setup"
    val repoUrl = "https://github.com/saki4510t/UVCCamera.git"
    val dest = file("TempUVCCamera")
    
    // Solo clona si no existe
    if (!dest.exists()) {
        commandLine("git", "clone", repoUrl, "TempUVCCamera")
    }
}

tasks.register<Copy>("copyLibUvcCamera") {
    dependsOn("cloneNuuneoiUvc")
    from("TempUVCCamera/libuvccamera")
    into("libuvccamera")
}
