from tools.label import LabelConverter


def test_label():
    y_text = ('Z25주2306', 'J64구8749', 'Z77누4381', 'C84버1051',
              'Z22주2222', 'A00주0000')
    label = LabelConverter('label.txt', 8)
    y_label, seq = label.to_tensor(y_text)

    decoded_text = label.to_text(y_label)
    for t1, t2 in zip(y_text, decoded_text):
        assert t1 == t2
