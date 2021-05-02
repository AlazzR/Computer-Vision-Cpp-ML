cmake .. -G "Visual Studio 16 2019" -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -T host=x64

cmake --build . --config Release

./mini-projects.exe

https://www.kaggle.com/uciml/pima-indians-diabetes-database
