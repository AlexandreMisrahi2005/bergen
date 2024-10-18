'''
Utility functions for dataset processors
'''

def chunk_text(text, id, title=None, max_size=1000, overlap=200, words_or_chars='chars'):
    """
    Chunk the given text into parts with a maximum size and overlap, prepending the title to each chunk.
    
    Args:
        text (str): The text to chunk.
        id (str): The id of the text.
        title (str): The title of the text. If None, no title is prepended.
        max_size (int): The maximum size of each chunk. Can be for characters or words.
        overlap (int): The overlap between chunks. Can be for characters or words.
        words_or_chars (str): Whether to chunk by characters ('chars') or words ('words'). Default is 'chars'.

    Returns:
        List[Dict[str, str]]: list of chunks and ids.
    """
    if title is None:
        title = ""
    if words_or_chars == 'words':
        text = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + max_size
        if start + overlap >= len(text):
            break
        chunk = ' '.join(text[start:end]) if words_or_chars == 'words' else text[start:end]
        chunk = title + ": " + chunk  # Prepend the title
        chunks.append({'id': f"{id}_{chunk_id}", 'content': chunk})
        start = end - overlap
        chunk_id += 1

    return chunks

def listify_label(row):
    row['label'] = [row['label']]
    return row

if __name__ == "__main__":
    text = "This is a test text to chunk into smaller parts. Once upon a time in a land far far away, there was a princess who lived in a castle. The princess had a pet dragon who was very friendly and loved to play with the princess. The princess and the dragon would go on many adventures together, exploring the enchanted forest and the magical mountains. One day, the princess and the dragon stumbled upon a hidden cave filled with treasure. The princess and the dragon were overjoyed and decided to share the treasure with the people of the kingdom. And they all lived happily ever after. This is the end of the test text."
    chunks = chunk_text(text, id='test', title='Test Title', max_size=10, overlap=2, words_or_chars='words')
    print(chunks)
    chunks = chunk_text(text, id='test', title='Test Title', max_size=100, overlap=20, words_or_chars='chars')
    print(chunks)