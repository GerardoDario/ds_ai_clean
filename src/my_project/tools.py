# src/my_project/tools.py

from pydantic_ai import tool

@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números."""
    return a + b

@tool
def contar_palabras(texto: str) -> int:
    """Cuenta palabras en un texto."""
    return len(texto.split())

@tool
def es_palindromo(texto: str) -> bool:
    """Revisa si una palabra o frase es palíndromo."""
    t = texto.lower().replace(" ", "")
    return t == t[::-1]
