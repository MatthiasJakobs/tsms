def get_sax_postfix(sax_alphabet_size):
    if sax_alphabet_size is None:
        return 'real'
    else:
        return f'quant{sax_alphabet_size}'
