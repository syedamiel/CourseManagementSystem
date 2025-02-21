import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Data placeholders for simplicity
users = []  # List of all users
courses = []  # List of all courses
enrollments = []  # List of course enrollments

# Helper function to generate unique user IDs
def generate_user_id():
    return len(users) + 1

# Classes
class User:
    def __init__(self, name, email):
        self.user_id = generate_user_id()
        self.name = name
        self.email = email

class Instructor(User):
    def __init__(self, name, email):
        super().__init__(name, email)
        self.bio = ""
        self.specialization = ""
        self.courses = []

    def create_course(self, title, description, category, price):
        course_id = len(courses) + 1
        new_course = Course(course_id, title, description, category, price, self)
        courses.append(new_course)
        self.courses.append(new_course)

    def manage_courses(self):
        for idx, course in enumerate(self.courses, 1):
            print(f"{idx}. {course.title} - {course.description} (${course.price})")
        choice = int(input("Select a course to edit (0 to cancel): "))
        if 0 < choice <= len(self.courses):
            course = self.courses[choice - 1]
            course.title = input(f"Enter new title (current: {course.title}): ") or course.title
            course.description = input(f"Enter new description (current: {course.description}): ") or course.description
            new_price = input(f"Enter new price (current: {course.price}): ")
            if new_price:
                course.price = float(new_price)
            print("Course updated successfully!")

class Student(User):
    def __init__(self, name, email):
        super().__init__(name, email)
        self.preferred_categories = []
        self.enrolled_courses = []
        self.completed_courses = []

    def choose_preferred_categories(self):
        print("Available categories:")
        for idx, category in enumerate(categories, 1):
            print(f"{idx}. {category.name} ({category.level})")
        category_choices = input("Select categories (comma-separated, 0 to cancel): ").split(',')
        for choice in category_choices:
            if choice.strip().isdigit():
                idx = int(choice.strip())
                if 0 < idx <= len(categories):
                    self.preferred_categories.append(categories[idx - 1])
        print("Preferred categories updated.")

    def browse_courses(self):
        print("Available courses:")
        for idx, course in enumerate(courses, 1):
            print(f"{idx}. {course.title} - {course.description} (${course.price})")
        choice = int(input("Select a course to view details (0 to cancel): "))
        if 0 < choice <= len(courses):
            course = courses[choice - 1]
            print(f"Title: {course.title}\nDescription: {course.description}\nPrice: ${course.price}\n")
            if course not in self.enrolled_courses:
                enroll_choice = input("Do you want to enroll in this course? (yes/no): ").strip().lower()
                if enroll_choice == "yes":
                    self.enrolled_courses.append(course)
                    course.add_student(self)
                    print("You have successfully enrolled in the course.")
            else:
                print("You are already enrolled in this course.")

    def view_enrolled_courses(self):
        print("Enrolled courses:")
        for idx, course in enumerate(self.enrolled_courses, 1):
            print(f"{idx}. {course.title}")

    def recommend_courses(self):
        recommendations = recommend_courses(self)
        print("Recommended courses:")
        if recommendations:
            for course in recommendations:
                print(f"- {course.title}: {course.description}")
        else:
            print("No recommendations available at the moment.")

    def mark_course_completed(self):
        print("Enrolled courses:")
        for idx, course in enumerate(self.enrolled_courses, 1):
            print(f"{idx}. {course.title}")
        choice = int(input("Select a course to mark as completed (0 to cancel): "))
        if 0 < choice <= len(self.enrolled_courses):
            course = self.enrolled_courses.pop(choice - 1)
            self.completed_courses.append({"course": course, "rating": None, "feedback": None})
            print(f"Marked '{course.title}' as completed.")

    def give_feedback_on_completed_courses(self):
        print("Completed courses:")
        for idx, entry in enumerate(self.completed_courses, 1):
            course = entry["course"]
            print(f"{idx}. {course.title}")
        choice = int(input("Select a course to give feedback (0 to cancel): "))
        if 0 < choice <= len(self.completed_courses):
            entry = self.completed_courses[choice - 1]
            course = entry["course"]
            rating = float(input(f"Rate the course '{course.title}' (1-5): "))
            feedback = input("Leave your feedback: ")
            entry["rating"] = rating
            entry["feedback"] = feedback
            course.leave_review(rating, feedback)

class Course:
    def __init__(self, course_id, title, description, category, price, instructor):
        self.course_id = course_id
        self.title = title
        self.description = description
        self.category = category
        self.price = price
        self.instructor = instructor
        self.rating = 0
        self.enrolled_students = []
        self.feedback = []

    def add_student(self, student):
        self.enrolled_students.append(student)

    def leave_review(self, rating, feedback):
        self.rating = (self.rating + rating) / 2 if self.rating else rating
        self.feedback.append(feedback)
        print(f"Thank you for your feedback on '{self.title}'.")

class Category:
    def __init__(self, category_id, name, level):
        self.category_id = category_id
        self.name = name
        self.level = level

#Recommendation system
def recommend_courses(student):
    if not student.completed_courses:
        print("You have no completed courses to base recommendations on.")
        return []

    # Gather completed categories from the 'course' key in dictionaries
    completed_categories = {entry["course"].category for entry in student.completed_courses}

    # Content-Based Filtering
    course_descriptions = [course.description for course in courses]
    tfidf = TfidfVectorizer().fit_transform(course_descriptions)
    description_similarities = cosine_similarity(tfidf)

    # Collaborative Filtering
    user_courses = np.zeros((len(users), len(courses)))
    for enrollment in enrollments:
        user_courses[enrollment['user_id'] - 1, enrollment['course_id'] - 1] = 1
    user_similarity = cosine_similarity(user_courses)

    # Combine Scores
    combined_scores = np.zeros(len(courses))
    for completed_entry in student.completed_courses:
        completed_idx = completed_entry["course"].course_id - 1
        combined_scores += description_similarities[completed_idx]

    # Get similar users
    student_idx = student.user_id - 1
    similar_users = user_similarity[student_idx]
    for other_student_idx, similarity in enumerate(similar_users):
        if other_student_idx != student_idx:
            for course_idx in range(len(courses)):
                combined_scores[course_idx] += similarity * user_courses[other_student_idx, course_idx]

    # Exclude already completed/enrolled courses and filter by category
    enrolled_and_completed_ids = {entry["course"].course_id for entry in student.completed_courses + student.enrolled_courses}
    recommended_courses = [
        courses[idx] for idx in np.argsort(combined_scores)[::-1]
        if (idx + 1) not in enrolled_and_completed_ids
        and combined_scores[idx] > 0
        and courses[idx].category in completed_categories  # Match category
    ]

    if not recommended_courses:
        print("No recommendations available based on your preferences.")
        return []

    return recommended_courses



def visualize_course_ratings():
    if courses:
        course_names = [course.title for course in courses]
        course_ratings = [course.rating for course in courses]
        plt.bar(course_names, course_ratings, color='skyblue')
        plt.xlabel("Courses")
        plt.ylabel("Average Rating")
        plt.title("Course Ratings Visualization")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No courses available to visualize.")

# Admin Console
def admin_console():
    while True:
        print("\n--- Admin Console ---")
        print("1. View enrollment reports")
        print("2. View leaderboard")
        print("3. System maintenance")
        print("4. Logout")

        choice = input("Enter your choice: ")
        if choice == "1":
            print("\n--- Enrollment Reports ---")
            for course in courses:
                print(f"{course.title}: {len(course.enrolled_students)} students enrolled")
        elif choice == "2":
            print("\n--- Leaderboard ---")
            top_courses = sorted(courses, key=lambda c: c.rating, reverse=True)[:5]
            for idx, course in enumerate(top_courses, 1):
                print(f"{idx}. {course.title} - {course.rating:.1f} stars")
        elif choice == "3":
            print("\nSystem maintenance is currently underway. Please wait...")
        elif choice == "4":
            print("Logging out of admin console...")
            break
        else:
            print("Invalid choice. Please try again.")

# Main Menu
def main_menu():
    while True:
        try:
            print("\n--- Course Management System ---")
            print("1. Register")
            print("2. Login to Dashboard")
            print("3. Admin Console")
            print("4. Exit")

            choice = input("Enter your choice: ")
            if not choice.isdigit():
                print("Invalid input. Please enter a number between 1 and 4.")
                continue

            choice = int(choice)

            if choice == 1:
                register_user()
            elif choice == 2:
                login_to_dashboard()
            elif choice == 3:
                key = input("Enter admin key: ")
                if key == "admin":
                    admin_console()
                else:
                    print("Invalid admin key.")
            elif choice == 4:
                print("Exiting... Goodbye!")
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting... Goodbye!")
            break

# User Registration
def register_user():
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    print("Choose your role:")
    print("1. Student")
    print("2. Instructor")
    role_choice = input("Enter your choice: ")

    if role_choice == "1":
        user = Student(name, email)
        users.append(user)
        print(f"Student account created. Your User ID: {user.user_id}")
    elif role_choice == "2":
        user = Instructor(name, email)
        users.append(user)
        print(f"Instructor account created. Your User ID: {user.user_id}")
    else:
        print("Invalid role selected.")

# Login to Dashboard
def login_to_dashboard():
    user_id = input("Enter your User ID: ")
    user = next((u for u in users if str(u.user_id) == user_id), None)

    if user:
        if isinstance(user, Student):
            student_dashboard(user)
        elif isinstance(user, Instructor):
            instructor_dashboard(user)
    else:
        print("User not found. Please register first.")

# Student Dashboard
def student_dashboard(student):
    while True:
        print("\n--- Student Dashboard ---")
        print("1. Choose preferred categories")
        print("2. Browse courses")
        print("3. View enrolled courses")
        print("4. View recommended courses")
        print("5. View completed courses")
        print("6. Give feedback on completed courses")
        print("7. Visualize course ratings")
        print("8. Logout")

        choice = input("Enter your choice: ")
        if choice == "1":
            student.choose_preferred_categories()
        elif choice == "2":
            student.browse_courses()
        elif choice == "3":
            student.view_enrolled_courses()
        elif choice == "4":
            student.recommend_courses()
        elif choice == "5":
            student.mark_course_completed()
            print("Completed courses:")
            for entry in student.completed_courses:
                for course in student.completed_courses:
                    course = entry["course"]
                    print(f"- {course.title}")
                    rating = entry["rating"]
                    feedback = entry["feedback"]
                    print(
                        f"- {course.title} (Rating: {rating if rating else 'N/A'}, Feedback: {feedback if feedback else 'N/A'})")
        elif choice == "6":
            student.give_feedback_on_completed_courses()
        elif choice == "7":
            visualize_course_ratings()
        elif choice == "8":
            print("Logging out of student dashboard...")
            break
        else:
            print("Invalid choice. Please try again.")

# Instructor Dashboard
def instructor_dashboard(instructor):
    while True:
        print("\n--- Instructor Dashboard ---")
        print("1. Insert/Update Bio")
        print("2. Insert/Update Specialization")
        print("3. Create a new course")
        print("4. Manage courses")
        print("5. View student feedback")
        print("6. Visualize course ratings")
        print("7. Logout")

        choice = input("Enter your choice: ")
        if choice == "1":
            instructor.bio = input("Enter your bio: ")
            print("Bio updated.")
        elif choice == "2":
            instructor.specialization = input("Enter your specialization: ")
            print("Specialization updated.")
        elif choice == "3":
            title = input("Enter course title: ")
            description = input("Enter course description: ")
            print("Available categories:")
            for idx, category in enumerate(categories, 1):
                print(f"{idx}. {category.name} ({category.level})")
            category_choice = int(input("Select a category (0 to cancel): "))
            if 0 < category_choice <= len(categories):
                category = categories[category_choice - 1]
                price = float(input("Enter course price: "))
                instructor.create_course(title, description, category, price)
        elif choice == "4":
            instructor.manage_courses()
        elif choice == "5":
            for course in instructor.courses:
                print(f"Course: {course.title}")
                for feedback in course.feedback:
                    print(f"- {feedback}")
        elif choice == "6":
            visualize_course_ratings()
        elif choice == "7":
            print("Logging out of instructor dashboard...")
            break
        else:
            print("Invalid choice. Please try again.")

# Pre-defined categories
categories = [
    Category(1, "Programming", "Beginner"),
    Category(2, "Programming", "Intermediate"),
    Category(3, "Programming", "Advanced"),
    Category(4, "Data Science", "Beginner"),
    Category(5, "Data Science", "Intermediate"),
    Category(6, "Data Science", "Advanced"),
    Category(7, "Cyber Security", "Beginner"),
    Category(8, "Cyber Security", "Intermediate"),
    Category(9, "Cyber Security", "Advanced")
]

# Pre-existing courses
courses = [
    Course(1, "Python Basics", "Learn Python from scratch.", categories[0], 49.99, None),
    Course(2, "Intermediate Data Science", "Learn intermediate Data Science.", categories[4], 99.99, None),
    Course(3, "Advanced Cyber Security", "Protect systems and data.", categories[8], 149.99, None),
    Course(4, "Java Basics", "Learn Java from scrath", categories[0], 59.99, None),
    Course(5, "Introduction to Javascript", "Introductory course to Javascript", categories[0], 39.99, None),
    Course(6, "Advanced Cyber Security", "Protect systems and data.", categories[4], 149.99, None),
]

# Run the program
if __name__ == "__main__":
    main_menu()
