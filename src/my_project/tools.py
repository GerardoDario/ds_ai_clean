# src/my_project/tools.py

def sumar(a: int, b: int) -> int:
    """Suma dos números."""
    return a + b

def contar_palabras(texto: str) -> int:
    """Cuenta palabras en un texto."""
    return len(texto.split())

def es_palindromo(texto: str) -> bool:
    """Revisa si una palabra o frase es palíndromo."""
    t = texto.lower().replace(" ", "")
    return t == t[::-1]
