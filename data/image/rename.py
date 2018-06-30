#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
修改imagelist.txt中的图片名，增加前缀
'''

input_file_name = "imagelist.txt"
output_file_name = "imagelist_new.txt"

if __name__ == '__main__':
    input_file = open(input_file_name, 'r')
    output_file = open(output_file_name, 'w')

    for l in input_file:
        l = "image/"+l
        output_file.write(l)

    input_file.close()
    output_file.close()
