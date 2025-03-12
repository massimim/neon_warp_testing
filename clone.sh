set -e
set -x

# Check if the 'ssh' argument is passed
USE_SSH=false
if [ "$1" == "ssh" ]; then
  USE_SSH=true
fi

# Set the repository URLs based on the argument
if [ "$USE_SSH" == true ]; then
  NEON_REPO="git@github.com:Autodesk/Neon.git"
  # WARP_REPO="git@github.com:massimim/warp.git"
  WARP_REPO="https://github.com/massimim/warp.git"
else
  NEON_REPO="https://github.com/Autodesk/Neon.git"
  WARP_REPO="https://github.com/massimim/warp.git"
fi

rm -fr neon
rm -fr warp

git clone $NEON_REPO -b py-local-src neon
git clone $WARP_REPO -b external-source-support warp

