from typing import Any, Dict, List

from .format import format_prompt_for_llm


class StoryPrompt:
    system_msg = "You are a helpful assistant for aiding in science experiments."

    @staticmethod
    def project_title(
            field: str, 
            domain: str, 
            subdomain: str, 
            subdomain_description: str, 
            num_project_titles: int) -> List[Dict[str, Any]]:
        
        field = field.replace("_", " ")
        user_msg = (
            "Please help me come up with some ideas for new research projects. "
            f'The projects should relate to "{domain}", specifically "{subdomain}". '
            f"Here's a description of the research area: {subdomain_description}.\n\n"
            "**These should be realistic projects that current researchers at a university or industry lab might be working on right now.** "
            "The projects should require data that is empirical; they may be theoretical, but they should always have empirical experiments as well. "
            "The project should also depend on data that can be encoded as a flat table (i.e. no images, massive vectors/matrices, etc.). "
            "There should be a set of independent variables and a set of dependent, measured variables that answer the research question. "
            "Consider the field's open questions, what the current state of the community's progress towards those questions looks like, and where there might be an opportunity for new research projects.\n\n"
            "Please generate the project ideas as a list of potential project names. "
            f"Specifically, please generate {num_project_titles} project names. "
            "The project names should be concise and descriptive, suitable for a scientific research project.\n\n"
            'Format your response as a JSON list of dictionaries, where each dictionary contains a "title" that maps to a string with the project\'s name, and a "repository_name" that maps to an appropriate project root directory name, like you might find on Github. '
            "Think out loud for a bit before responding with your decision."
        )
        return format_prompt_for_llm(user_msg, StoryPrompt.system_msg)
    
    @staticmethod
    def description(subdomain: str, project_title: str) -> List[Dict[str, Any]]:
        user_msg = (
            "Generate a description for a scientific research project. "
            f'The project is in the area "{subdomain}", and is called "{project_title}". '
            "This should be a realistic project that a current researcher at a university or industry lab might be conducting right now.\n\n"
            "Please follow these guidelines:\n\n"
            "1. The project should require data that is empirical; it may be theoretical, but it should always have empirical experiments as well.\n"
            "2. The project should also depend on data that can be encoded as a flat table (i.e. no images, massive vectors/matrices, etc.).\n"
            "3. There should be a set of independent variables and a set of dependent, measured variables that answer the research question. "
            "Do your best to come up with an excaustive list of any independent and dependent variables that might be relevant, as well as possible confounders.\n"
            "4. The description should provide context about the project, its goals, hypotheses, and the kinds of data collected.\n"
            "5. Discuss what each file represents (e.g. observations of a specific mouse, a particular neural network configuration, etc.), and what each row in a file represents. "
            "Consider what variables might be best suited for the directory/file-path structure, and what variables makes the most sense to encode as individual columns in the file. "
            "Don't provide a specific directory/file-path structure yet, but discuss ideas for where different information might go w.r.t. directories, file names, and in the files themselves.\n"
            "6. Explain the significance of the project in its field, and why it's novel and note-worthy.\n\n"
            'Format your response as JSON with a single key "description" that maps to a string with prose addressing the instructions above. '
            "Think out loud for a bit before responding with your decision."
        )
        return format_prompt_for_llm(user_msg, StoryPrompt.system_msg)
    
    @staticmethod
    def abstract(project_title: str, description: str) -> List[Dict[str, Any]]:
        user_msg = (
            f'Given a research project called "{project_title}" with the following description:\n\n{description}\n\n'
            "Generate a one-paragraph abstract (c. 6-7 sentences, 150-250 words) like you might find in an academic paper. "
            "The goal is to provide a concise, accurate, and self-contained summary of the research paper, allowing readers to quickly determine its relevance to their interests. "
            'Format as JSON with a single key "abstract" that maps to a string with prose containing your abstract. '
            "Think out loud for a bit before responding with your decision."
        )
        return format_prompt_for_llm(user_msg, StoryPrompt.system_msg)
