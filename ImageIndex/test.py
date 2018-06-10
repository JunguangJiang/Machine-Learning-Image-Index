from ImageIndex.test_module import get_prefix
def write():
    print(get_prefix())
    with open("Hello","w") as f:
        f.write("Hello")

if __name__ == '__main__':
    write()