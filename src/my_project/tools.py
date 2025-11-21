# src/my_project/tools.py
from pypdf import PdfReader

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

def leer_pdf(path: str) -> str:
    """
    Lee un PDF del disco y retorna su contenido como texto.
    """
    reader = PdfReader(path)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n".join(pages_text)
