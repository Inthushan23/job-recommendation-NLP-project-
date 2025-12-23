from fastapi import APIRouter
from fastapi import Request


router = APIRouter(prefix="/recommender")


@router.get("/get-df")
def get_df(request: Request):
    recommender = request.app.state.recommender
    
    tastes_df = recommender.tastes_df
    questions_df = recommender.questions_df
    skills_df = recommender.skills_df
    print(tastes_df)

    return {"tastes": tastes_df.to_dict(orient = "records"), 
            "questions": questions_df.to_dict(orient = "records"), 
            "skills": skills_df.to_dict(orient = "records")
            }


@router.post("/get-domain")
def get_domain(input1, input2, request: Request):
    recommender = request.app.state.recommender
    encoder = request.app.state.encoder
    tastes_df, domain = recommender.get_domain(input1, input2, encoder)
    return {"tastes_df": tastes_df.to_dict(orient = "list"), "domain": domain}

@router.post("/question-based-analysis")
def question_based_analysis(domain, input3, input4, input5, request: Request):
    recommender = request.app.state.recommender
    encoder = request.app.state.encoder

    jobs_competencies, skills_competency = recommender.question_based_sim(domain, input3, input4, input5, encoder)

    return {"jobs_competencies": jobs_competencies.to_dict(orient = "list"), "skills_competency": skills_competency.to_dict(orient="list")}
