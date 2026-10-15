"""Decode stock wire representations within the benchmark's compact-JSON domain."""
import json
import string


def quill_json_text(text):
    """Invert Quill printable-char hex escapes, preserving literal JSON \\x text.

    Stock Quill leaves printable backslashes unchanged. Valid JSON contains
    literal backslashes in pairs; an odd backslash run before xHH therefore
    denotes one byte introduced by its sanitizer. This inverse is specific to
    valid JSON text, not a claimed general inverse for arbitrary log strings.
    """
    if '\\x' not in text:return text
    result=bytearray();i=0;converted=False
    while i<len(text):
        if text[i]!='\\':
            result.extend(text[i].encode());i+=1;continue
        j=i
        while j<len(text) and text[j]=='\\':j+=1
        count=j-i
        if count%2 and j+2<len(text) and text[j]=='x' and all(c in string.hexdigits for c in text[j+1:j+3]):
            result.extend(b'\\'*(count-1));result.append(int(text[j+1:j+3],16));i=j+3;converted=True
        else:
            result.extend(b'\\'*count);i=j
    decoded=result.decode('utf-8')
    if converted:json.loads(decoded)  # Validate the domain used to disambiguate.
    return decoded
