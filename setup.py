from setuptools import find_packages, setup

setup(
    name='MEDICAL-CHATBOT',
    version='0.0.1',
    author='zunaira',
    author_email='zunairanoreen127@gail.com',
    install_requires=[
        "sentence-transformers",
        "langchain",
        "pypdf",
        "flask",
        "python-dotenv",
        "langchain-pinecone",
        "langchain-community",
        "langchain-experimental",
        "langchain-openai",
        "langchain-groq",
        "groq",
        'pinecone[grpc]',
    ],
    packages=find_packages()
)