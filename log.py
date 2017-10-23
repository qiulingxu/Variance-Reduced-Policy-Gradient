class log:
    def __init__(self,taskname):
        self.filename=taskname+".txt"
        self.fp=open(self.filename,'ab')
    def write(self,content):
        self.fp.write(content)
        self.fp.flush()

