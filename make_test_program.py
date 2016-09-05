
import argparse
import os

parser = argparse.ArgumentParser(description="program arguments")
parser.add_argument('--src', type=str, action="store", dest="src")
args = parser.parse_args()

name = args.src
gox = name
src = name + ".cpp"

cmake = """
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(%(gox)s %(src)s)
target_link_libraries(%(gox)s core ${ARMADILLO_LIBRARIES}) 
"""   % locals()

cmd = """
mkdir tests/%(name)s 
cp %(src)s tests/%(name)s
cat << EOF > tests/%(name)s/CMakeLists.txt
%(cmake)s
EOF
""" % locals()

print cmake
print cmd

#os.system(cmd)
