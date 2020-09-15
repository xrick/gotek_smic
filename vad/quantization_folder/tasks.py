import cffi
import invoke
import pathlib

def clean(c):
    """ Remove any built objects """
    for pattern in ["*.o", "*.so", "cffi_example* cython_wrapper.cpp"]:
        c.run("rm -rf {}".format(pattern))

def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))

@invoke.task
def build_fft(c):
    """ Build the shared library for the sample C code """
    print_banner("Building C Library")
    invoke.run("gcc -c -Wall -Werror -fpic libfft.c -I /home/user/.conda/envs/pytorchenv151/bin")
    invoke.run("gcc -shared -o libfft.so libfft.o")
    print("* Complete")

@invoke.task(build_fft)
def build_cffi(c):
    """ Build the CFFI Python bindings """
    print_banner("Building CFFI Module of FFT")
    ffi = cffi.FFI()

    this_dir = pathlib.Path().absolute()
    h_file_name = this_dir / "libfft.h"
    with open(h_file_name) as h_file:
        ffi.cdef(h_file.read())

    ffi.set_source(
        "c_fft",
        # Since we are calling a fully built library directly no custom source
        # is necessary. We need to include the .h files, though, because behind
        # the scenes cffi generates a .c file which contains a Python-friendly
        # wrapper around each of the functions.
        '#include "libfft.h"',
        # The important thing is to include the pre-built lib in the list of
        # libraries we are linking against:
        libraries=["fft"],
        library_dirs=[this_dir.as_posix()],
        extra_link_args=["-Wl,-rpath,."],
    )

    ffi.compile()
    print("* Complete")


@invoke.task()
def test_fft(c):
    """ Run the script to test CFFI """
    print_banner("Testing CFFI Module")
    invoke.run("python3 cffi_test.py", pty=True)