pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        jcenter() // Added jcenter
        maven { url = uri("https://jitpack.io") }
        maven { url = uri("https://raw.github.com/saki4510t/libcommon/master/repository/") } // Added saki4510t
        maven { url = uri("https://gitee.com/liuchaoya/libcommon/raw/master/repository/") } // Added gitee.com
    }
}

rootProject.name = "MyApplication3"
include(":app")
include(":libuvccamera")
