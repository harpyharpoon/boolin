import base64
import binascii
import zlib

# Function to read binary data from a text file
def read_binary_data(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    return data

# Function to convert binary data to hexadecimal
def to_hex(data):
    return data.hex()

# Function to convert binary data to base64
def to_base64(data):
    return base64.b64encode(data).decode('utf-8')

# Function to try different character encodings
def try_decoding(data):
    encodings = ['utf-8', 'ascii', 'latin-1', 'utf-16']
    decoded_data = {}
    for encoding in encodings:
        try:
            decoded_data[encoding] = data.decode(encoding)
        except UnicodeDecodeError:
            decoded_data[encoding] = None
    return decoded_data

# Function to decompress data using zlib
def decompress_data(data):
    try:
        return zlib.decompress(data)
    except zlib.error:
        return None

# Function to identify file type based on common file signatures
def identify_file_type(data):
    file_signatures = {
        b'\xFF\xD8\xFF': 'JPEG image',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'PNG image',
        b'\x47\x49\x46\x38': 'GIF image',
        b'\x25\x50\x44\x46': 'PDF document',
        b'\x50\x4B\x03\x04': 'ZIP archive',
        # Add more signatures as needed
    }
    for signature, file_type in file_signatures.items():
        if data.startswith(signature):
            return file_type
    return 'Unknown file type'

# Function to save data to a file
def save_to_file(file_name, data):
    with open(file_name, 'w') as file:
        file.write(data)

# Main function to decode binary data
def main():
    file_path = '/home/harpoon/Documents/steg/secret.txt'  # Replace with your file path
    binary_data = read_binary_data(file_path)
    
    # Convert to hexadecimal
    hex_representation = to_hex(binary_data)
    save_to_file('hex_representation.txt', hex_representation)
    
    # Convert to base64
    base64_representation = to_base64(binary_data)
    save_to_file('base64_representation.txt', base64_representation)
    
    # Try different character encodings
    decoded_data = try_decoding(binary_data)
    for encoding, decoded in decoded_data.items():
        if decoded:
            save_to_file(f'decoded_{encoding}.txt', decoded)
    
    # Attempt to decompress data
    decompressed_data = decompress_data(binary_data)
    if decompressed_data:
        save_to_file('decompressed_data.txt', decompressed_data.decode('utf-8', errors='ignore'))
    
    # Identify file type
    file_type = identify_file_type(binary_data)
    print(f'Identified File Type: {file_type}\n')

if __name__ == "__main__":
    main()