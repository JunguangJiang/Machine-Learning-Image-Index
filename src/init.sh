PATH_TO_PROTOC=`pwd`

# First, install slim's "nets" package.
cd slim
sudo pip install -e .

# Second, setup the object_detection module by editing PYTHONPATH.
cd ..
# From src
delf/protoc-3/bin/protoc object_detection/protos/*.proto --python_out=.

cd delf
# From src/delf/
${PATH_TO_PROTOC?}/delf/protoc-3/bin/protoc delf/protos/*.proto --python_out=.

# From src/delf/
sudo pip install -e . # Install "delf" package.