class Account:
    def __init__(self, id):
        self.id = id
        id = 666
        print("Constructor self", self.id)
        print("Constructor id", id)

acc = Account(123)
print(acc.id)
