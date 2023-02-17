import os
import shutil


class ToolBags:
    def __init__(self) -> None:
        self.res_ = []

    @staticmethod
    def addRst(Path: str, FileName: str):  # 新建一个rst文件
        isExists = os.path.exists(Path)
        if not isExists:
            File = open('{}/{}'.format(Path, FileName), 'w', encoding='utf-8')
            print("INFO::" + Path + ' 创建{}成功'.format(Path + FileName))
            return File
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print("INFO::" + Path + ' {}文件已存在'.format(Path + FileName))
            return False

    @staticmethod
    def addFolder(path: str):  # 新建一个文件夹
        isExists = os.path.exists(path)
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
            print("INFO::" + path + ' 创建{}成功'.format(path))
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print("INFO::" + path + ' {}目录已存在'.format(path))
            return False

    @staticmethod
    def ls(path: str):  # 获得文件夹下的子文件和文件夹的名字的列表
        x = os.listdir(path)
        return x

    @staticmethod
    def is_dir(Route):
        return os.path.isdir(Route)

    @staticmethod
    def copyfile(srcfile, dstpath):  # 复制函数
        # srcfile 需要复制、移动的文件
        # dstpath 目的地址
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)  # 创建路径
            shutil.copy(srcfile, dstpath + fname)  # 复制文件
            print("copy %s -> %s" % (srcfile, dstpath + fname))

    def Copy_from_anything(self, srcPath: str, dstPath: str):
        # 将srcPath文件或者地址下的所有文件赋值到dstPath地址以下
        isFile_src = os.path.isfile(srcPath)
        isdir_src = os.path.isdir(srcPath)
        if not (isFile_src or isdir_src):
            print("不存在文件或目录:{}".format(srcPath))
            return
        isFile_dst = os.path.isfile(dstPath)
        isdir_dst = os.path.isdir(dstPath)
        if not (isFile_dst or isdir_dst):
            print("不存在目录:{}".format(srcPath))
            os.makedirs(dstPath)  # 创建路径
            print("创建目录:{}".format(dstPath))
        if isFile_src:
            self.copyfile(srcPath, dstPath + '/')
        else:  # 从目录到目录
            FileList = self.getAllFilesUnder(srcPath)
            for fileI in FileList:
                newDstPath = dstPath + str(fileI).replace(srcPath, "")
                if os.path.isdir(fileI):
                    os.makedirs(newDstPath)
                else:
                    fpath, fname = os.path.split(newDstPath)
                    self.copyfile(fileI, newDstPath.replace(fname, ''))

    def getAllFilesUnder(self, route_):
        self.res_ = []
        for filepath, dirnames, filenames in os.walk(route_):
            for filename in filenames:
                path__ = os.path.join(filepath, filename)
                self.res_.append(path__)
        return self.res_


if __name__ == '__main__':
    a = ToolBags()
    route1 = "/Users/andrewlee/Desktop/Projects/代码复用库/labs/file1/hello.cpp"
    route2 = "/Users/andrewlee/Desktop/Projects/代码复用库/labs"
    z = a.Copy_from_anything(route1, route2)
    # print(a.is_dir("/Users/andrewlee/Desktop/PPSUC_WIKI_WEB/GitToSphinx/mkRST.py"))
