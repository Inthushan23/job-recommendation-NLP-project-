from services.data_service.src.repository.load_data import load_file


# Check if the files are loaded correctly
def test_load_files():
    tastes, questions, skills = load_file()

    assert tastes is not None 
    assert questions is not None 
    assert skills is not None 

    assert tastes.empty == False
    assert questions.empty == False
    assert skills.empty == False

