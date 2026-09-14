set -e
set -x

# Check if the 'ssh' argument is passed
USE_SSH=false
if [ "$1" == "ssh" ]; then
  USE_SSH=true
fi

# py-local-src is the branch the neon_gpu wheels are built from, and it pins
# Warp to massimim/warp@external-source-support-update4, which carries the BVH
# shared-stack fix Neon's 3D (4,4,4) thread blocks need.
NEON_BRANCH="py-local-src"
WARP_BRANCH="external-source-support-update4"

# Set the repository URLs based on the argument
if [ "$USE_SSH" == true ]; then
  NEON_REPO="git@github.com:Autodesk/Neon.git"
  WARP_REPO="git@github.com:massimim/warp.git"
else
  NEON_REPO="https://github.com/Autodesk/Neon.git"
  WARP_REPO="https://github.com/massimim/warp.git"
fi

rm -fr neon
rm -fr warp

git clone "$NEON_REPO" -b "$NEON_BRANCH" neon
git clone "$WARP_REPO" -b "$WARP_BRANCH" warp
