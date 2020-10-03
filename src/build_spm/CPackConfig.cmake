# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/home/9/16B02273/sentencepiece;/home/9/16B02273/Sentencepiece_py/src/build_spm")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Taku Kudo")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/apps/t3/sles12sp2/gsic/cmake/3.17.2/share/cmake-3.17/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "sentencepiece built using CMake")
set(CPACK_GENERATOR "7Z")
set(CPACK_INSTALL_CMAKE_PROJECTS "/home/9/16B02273/Sentencepiece_py/src/build_spm;sentencepiece;ALL;/")
set(CPACK_INSTALL_PREFIX "/usr/local")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "sentencepiece 0.1.93")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "sentencepiece 0.1.93")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OUTPUT_CONFIG_FILE "/home/9/16B02273/Sentencepiece_py/src/build_spm/CPackConfig.cmake")
set(CPACK_PACKAGE_CONTACT "taku@google.com")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/apps/t3/sles12sp2/gsic/cmake/3.17.2/share/cmake-3.17/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "sentencepiece built using CMake")
set(CPACK_PACKAGE_FILE_NAME "sentencepiece-0.1.93-Linux")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "sentencepiece 0.1.93")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "sentencepiece 0.1.93")
set(CPACK_PACKAGE_NAME "sentencepiece")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION "0.1.93")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "93")
set(CPACK_RESOURCE_FILE_LICENSE "/home/9/16B02273/sentencepiece/LICENSE")
set(CPACK_RESOURCE_FILE_README "/home/9/16B02273/sentencepiece/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "/apps/t3/sles12sp2/gsic/cmake/3.17.2/share/cmake-3.17/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TXZ")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;/dist/;/sdist/;~$;")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/9/16B02273/Sentencepiece_py/src/build_spm/CPackSourceConfig.cmake")
set(CPACK_STRIP_FILES "TRUE")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/9/16B02273/Sentencepiece_py/src/build_spm/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
