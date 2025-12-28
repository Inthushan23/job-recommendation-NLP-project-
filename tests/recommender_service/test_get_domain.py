from services.recommender_service.src.domain.recommender import Recommender, Encoder


recommender = Recommender()
encoder = Encoder()


def test_get_domain():
    assert (recommender.get_domain("I like Data and AI", "I do not like Finance", encoder)[1]  == "IT")
    assert (recommender.get_domain("I like helping people who have been injured. Giving them medicine.", "I do not like Finance", encoder)[1]  == "Medicine")
    assert (recommender.get_domain("I like money", "I do not like plants", encoder)[1]  == "Financial")
    assert (recommender.get_domain("I like to draw and paint", "I don't like science", encoder)[1]  == "Art")
    assert (recommender.get_domain("I like rockets and astrology.", "I don't like to draw", encoder)[1]  == "Space")
    assert (recommender.get_domain("I like animals, plants and gardening.", "I don't like finance", encoder)[1]  == "Nature")
    

def test_question_based_sim():
    jobs_competencies, skills_competency = recommender.question_based_sim("IT", "Data cleaning, data visualization", "I worked on a data visualization project using Power BI.", "SQL, DAX", encoder)

    assert jobs_competencies is not None
    assert (jobs_competencies.empty == False) 

    assert skills_competency is not None
    assert (skills_competency.empty == False) 

