from typing import TypedDict

# schema
class Review(TypedDict):
    sentiment : str
    prodcut_id : int
    summary : str

new_review: Review = {'sentiment':'negative', 'product_id':983, 'summary':'sdsadsdad'}

print(new_person)

# Example flow

# structured_model = model.with_structured_output(Review)

# result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse!.""")


# O/P - {'sentimment':positive, 'product_id':940, 'summary':'sdadsad'}

