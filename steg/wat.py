import base64

# Input string
encoded_text = "YJAQIWESQQZTYSCCYSPIJHIZFTISQNJXOJBZZILBQPIMGSFSIGZCZPLOZLLZSPQZDPQRQBBPZMJZSQQYZCCJIZOZIOEZGZAQBSQLGBSTEPDVORAKVAKYZXUCXRXEATCCAEWTWAITHNSPTTCSBZTCKCXTCSQHBUFKBREXTXETPROGHUWZIPQMCTOWEXCCITXPXQEWCIOSIRCRAEWWQWCFCFSETFDCPSXXXEODHMEEC"

# Check if it's valid Base64 by attempting to decode
try:
    decoded_text = base64.b64decode(encoded_text).decode("utf-8")
except Exception as e:
    decoded_text = str(e)  # Capture the error if decoding fails

decoded_text
