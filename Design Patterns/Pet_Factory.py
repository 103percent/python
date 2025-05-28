# Python code​​​​​​‌‌​‌​‌‌​​‌‌‌‌‌‌​‌‌​​‌​‌​‌ below
# Use print("messages...") to debug your solution.

show_expected_result = False
show_hints = False

class Pet:

	"""A simple pet class"""

	def __init__(self, name):
		self._name = name
		self._sound = "Silence"

	def __eq__(self, other):
		if isinstance(other, Pet):
			return (self._sound) == (other._sound)
		return NotImplemented

	def __str__(self):
		return '{} | {}'.format(self._name, self._sound)

	def speak(self):
		return self._sound

class Dog(Pet):

	"""A simple dog class"""

	def __init__(self, name):
		Pet.__init__(self, name)
		self._sound = "Woof!"

class Cat (Pet):

	"""A simple cat class"""

	def __init__(self, name):
		Pet.__init__(self, name)
		self._sound = "Meow!"

#Your Pig class code goes here

class Pig (Pet):

	"""A simple pig class"""

	def __init__(self, name):
		Pet.__init__(self, name)
		self._sound = "Oink!"



def get_pet(pet="dog"):

	"""The factory method"""

#Your code to create and add a pig goes here
	pets = dict(dog=Dog("Hope"), cat=Cat("Peace"), pig=Pig("Love"))

	return pets[pet]

def get_pig():
    return get_pet("pig")