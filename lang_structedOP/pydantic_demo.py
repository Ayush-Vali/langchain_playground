from pydantic import BaseModel, EmailStr, Field
from typing import Annotated, Optional

class Student(BaseModel):

    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')
    summary : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]


new_student = {'age':'32', 'email':'abc@gmail.com', 'summary':'nigga name'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

# student_json = student.model_dump_json()

print(student)