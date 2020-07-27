class patch_data():
    def __init__(self, num):
        self.num = num
        self.id = num * 10
    def say(self):
        print(self.num)
        print(self.id)


class data(patch_data):
    def __init__(self, num):
        super(data, self).__init__(num)
        self.idd = num*100

    def say(self):
        print(self.num)
        print(self.idd)
        print(self.id)


if __name__ == "__main__":
    pd = patch_data(5)
    pd.say()
    d = data(5)
    print()
    d.say()