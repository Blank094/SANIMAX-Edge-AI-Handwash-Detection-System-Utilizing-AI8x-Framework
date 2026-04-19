# This file can be used to set build configuration
# variables.  These variables are defined in a file called
# "Makefile" that is located next to this one.

# For instructions on how to use this system, see
# https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system

# **********************************************************
BOARD = FTHR_RevA
# Add your config here!

# Set a higher optimization level.  The increased performance
# is required for the CameraIF DMA code to work within the
# timing requirements of the Parallel Camera Interface.
MXC_OPTIMIZE_CFLAGS = -O2
PROJ_OPTIMIZE = -O2

PROJ_CFLAGS += -fdump-ipa-cgraph
PROJ_CFLAGS += -fstack-usage
PROJ_CFLAGS += -gdwarf-4