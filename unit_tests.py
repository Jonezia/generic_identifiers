from identifier_generifier import IdentifierGenerifier

def main():
	generifier = IdentifierGenerifier("python")

	source_code = """
	def foo():
		if bar:
			baz()
	"""

	print(generifier.generify(source_code))

	source_code = """
	def foo(bar):
		if bar:
			baz()
		if not bar:
			bar = 5
			beep()
		baz()
		bar = 2
	"""

	print(generifier.generify(source_code))



if __name__ == '__main__':
	main()
