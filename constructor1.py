#non-parameterized constructor

class Car:
      def __init__(self): #non-parameterized constructor
            self.color="white"
            self.speed=120

    def display(self):
        print("The color of the BMW is %s and the speed is %d" %(self.color,self.speed))


BMW =Car()
BMW.display()        