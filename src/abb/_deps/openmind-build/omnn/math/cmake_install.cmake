# Install script for directory: /home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-build/omnn/math/libmath.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so"
         OLD_RPATH "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-build/omnn/math:/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-build/omnn/rt:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmath.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Cache.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Constant.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/DuoValDescendant.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Exponentiation.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Formula.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/FormulaOfVaWithSingleIntegerRoot.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Fraction.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Infinity.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Integer.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Modulo.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/OpenOps.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Polyfit.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/PrincipalSurd.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Product.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/SequenceOrderComparator.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Sum.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/SySHA.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/SymmetricDouble.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/System.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Valuable.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/ValuableCollectionDescendantContract.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/ValuableDescendant.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/ValuableDescendantContract.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/VarHost.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/Variable.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/e.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/i.h"
    "/home/alesm512/ABB_New_Msc/src/abb/_deps/openmind-src/omnn/math/pi.h"
    )
endif()

