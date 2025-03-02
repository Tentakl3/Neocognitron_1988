import initStruct

if __name__ == '__main__':
    struct = initStruct.InitStruct()
    output = struct.generateMonotonic(base=0.9, size=20, planes=1, norm=True)
    struct.plotMonotonic(output, 20)