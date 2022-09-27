#!/bin/bash

echo 'get libs...'
mkdir install
mkdir install/models
rm *.tar
cp libai_core.hpp install/
cp build/libai_core.so install/
cp test_* install/
cp config.hpp install/
cp CMakeLists.txt install/
cp readme.md install/
cp /project/git/shawn_qian/objectdetection/rknn/*.rknn install/models
rm install/models/*_precompiled.rknn
ls install/
tar cf install.tar install
echo 'get libs done'

