import tree_sitter
import tree_sitter_python

class IdentifierGenerifier:
	def __init__(self, language: str):

		self.language = language
		if self.language == "python":
			self.parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
		else:
			raise ValueError("Invalid language")

		self.var_mapping = {}
		self.var_counter = 0

	def generify(self, code: str):
		"""
		Method to anonymize variable names in code passed as str

		Args:
			code (str): source code

		Returns:
			str: source code with identifier variables renamed
		"""
		try:
			code_bytes = code.encode('utf-8')
			tree = self.parser.parse(code_bytes)
			cursor = tree.walk()		
		except Exception as e:
			print(f"Error during Tree-sitter processing: {e}")

		# list of replacements [(start_byte, end_byte, new_identifier)]
		replacements = self._traverse_and_replace(cursor.node, code_bytes)

		new_code_bytes = self._replace_identifiers(code_bytes, replacements)
		return new_code_bytes.decode() 

	def _replace_identifiers(self, code_bytes, replacements):
		# perform replacements in reverse to avoid shifting byte positions
		for start, end, new_identifier in reversed(replacements):
			code_bytes = code_bytes[:start] + new_identifier.encode() + code_bytes[end:]
		return code_bytes

	def _traverse_and_replace(self, node, code_bytes):
		replacements = []
		if node.type == "identifier":
			identifier_name = code_bytes[node.start_byte:node.end_byte].decode()
			new_identifier = self._get_new_identifier(identifier_name)
			replacements.append((node.start_byte, node.end_byte, new_identifier))
		for child in node.children:
			replacements.extend(self._traverse_and_replace(child, code_bytes))
		return replacements

	def _get_identifier_type(self, node):
		# Basic identifier type
		if node.type != 'identifier':
			raise ValueError("this function should be called with identifier nodes")

		# Get parent node for context
		parent = node.parent
		if not parent:
			return None

		# Class definitions
		if parent.type == 'class_definition':
			return 'class'

		# Function definitions
		if parent.type == 'function_definition':
			# Check if this is a method (inside a class)
			current = parent
			while current and current.type != 'class_definition':
				current = current.parent
			return 'method' if current else 'function'

		# Parameters
		if parent.type in ['parameters', 'typed_parameter', 'default_parameter']:
			return 'parameter'

		# default to variable
		return 'variable'

	def _get_new_identifier(self, name: str):
		if name not in self.var_mapping:
			new_identifier = f"VAR_{self.var_counter}"
			self.var_mapping[name] = new_identifier
			self.var_counter += 1
		return self.var_mapping[name]	
