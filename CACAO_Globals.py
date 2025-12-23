'''
These are the global variables and classes used in the CACAO project.
They are used to store the configuration settings and other constants.
The variables are defined in a separate file to keep the code organized and maintainable.
The declarations here should rarely change from class to class.
'''

import pandas as pd
import numpy as np

from typing import Any
from scipy.stats import percentileofscore

import CACAO_AssignmentClass as AC

possible_versions = ["v1"] 

# Student information column names from CANVAS 
# The keys serve as possible "versions" that you can use, and this should be 
# stated in the specific class file. 

# Column 
student_name = {
    "v1" : ['Student'], # purposely a one-element list, will become a two element list after split
}

# Column 
student_id = {
    "v1" : 'SIS User ID',
}

# Row 
points_possible = {
    "v1" : '    Points Possible',
}


# There are some rows that are not students but rather contain class info or
# are placeholders. Remove these! These should be copy and pasted directly from 
# the CSV file to ensure that they are correct (which includes spacing).
row_removes = {
    "v1" : ['    Points Possible' , 'Student, Test'],
}



# Given one column where the data is "Last, First" this function will split it into two columns
def split_name_column(gbook: pd.DataFrame, col: str) -> None:
    # Split the column into two new columns
    gbook['Last'] = gbook[col].str.split(',').str[0].astype(str).str.strip()  # Ensure the last name is a string and strip whitespace
    gbook['First'] = gbook[col].str.split(',').str[1].astype(str).str.strip()  # Ensure the first name is a string and strip whitespace


    # Drop the original column
    gbook.drop(columns=[col], inplace=True)

    return None


# Possible ways to take the student name and make it into a last name, first name set of columns.
# If the columsn are already split, make this function do nothing. 
split_name = {
    "v1": split_name_column,
}


def get_category_items(gbook: pd.DataFrame, category_name: str) -> list[str]:
    
    for cat in gbook.class_assignment_categories.keys():
        if cat == category_name:
            return gbook.class_assignment_categories[category_name]['items'] + [ category_name + ' Total' ]  # return the items in the category plus the total column name


class Gradebook:

    def __init__(self, gbook_fname: str, class_metadata: dict[str,Any], 
        class_assignment_categories: dict[str,Any], global_versioned: dict[str,Any]) -> None:
        """
        Initialize the Gradebook object.
        :param gbook_fname: str, Path to the gradebook CSV file.
        :param class_metadata: dict[str, Any], Metadata for the class, including version and student info.
        :param class_assignment_categories: dict[str, Any], Assignment categories for the class.
        :param global_versioned: dict[str, Any], Global versioned settings for the program.
        """

        self.gbook_fname = gbook_fname
        assert isinstance(gbook_fname, str), "gbook_fname must be a str"
        self.class_metadata = class_metadata
        assert isinstance(class_metadata, dict), "class_metadata must be a dict"
        self.class_assignment_categories = class_assignment_categories
        assert isinstance(class_assignment_categories, dict), "class_assignment_categories must be a dict"
        self.global_versioned = global_versioned
        assert isinstance(global_versioned, dict), "global_versioned must be a dict"

        # Store version 
        self.version = class_metadata['version']
        assert self.version in possible_versions, f"Version {self.version} is not supported. Supported versions: {possible_versions}"

        # Read in the raw gradebook 
        self.raw_gbook = pd.read_csv(self.gbook_fname)
        self.gbook = self.raw_gbook.copy()

        # Create category class objects 
        self.assignment_categories = self._create_assignment_categories()

        # Check category point totals with class total 
        assert self.class_metadata['total_points'] == self._get_point_totals(), \
            f"Total points in metadata ({self.class_metadata['total_points']}) does not match calculated total points ({self._get_point_totals()})."
        print(f"Total course points: {self._get_point_totals()} points")

        # remove rows that are not students
        self._remove_rows()

        # Split names column, pop removes element from the list and returns it
        global_versioned["split_name"][self.version](self.gbook, self.global_versioned['student_name'][self.version].pop(0)) 
        self.global_versioned["student_name"][self.version].insert(0,'Last')
        self.global_versioned["student_name"][self.version].insert(0,'First')

        # Make sure student id colunn is integers 
        self.gbook[self.global_versioned['student_id'][self.version]] = self.gbook[self.global_versioned['student_id'][self.version]].astype(int)

        # Make gradebook now only student name, student id, and relevant columns
        self.gbook = self.gbook[self._get_relevant_columns()]


        print("Gradebook successfully initialized")



    def _create_assignment_categories_total_columns(self) -> None:
        ''' Create total columns for each assignment category in the gradebook. '''

        self.totals_columns = [] 

        for cat in self.assignment_categories:
            name = cat.name + ' Total'
            self.totals_columns.append(name)
            self.gbook[name] = 0.0

        return None


    def _get_relevant_columns(self) -> list[str]:
        relevant_columns = self.global_versioned['student_name'][self.version] + \
            [self.global_versioned['student_id'][self.version]] 
        for cat in self.assignment_categories:
            relevant_columns.extend(cat.get_column_names())

        return relevant_columns

    def _remove_rows(self) -> None:
        ''' Remove rows that are in the remove_rows list. '''

        for row in self.global_versioned["row_removes"][self.version]:
            #print(row)
            self.gbook = self.gbook[self.gbook.iloc[:,0] != row]

    def _get_point_totals(self) -> int:

        point_total = 0

        for cat in self.assignment_categories:
            point_total += cat.get_course_points()

        return point_total 

    def _create_assignment_categories(self) -> list[AC.AssignmentCategory]:

        cat_list=[]

        for key in self.class_assignment_categories.keys():
            print(f"Assignment category: {key} ({self.class_assignment_categories[key]["course_points"]} course points)")
            cat_list.append(AC.AssignmentCategory(gbook=self.raw_gbook, global_versioned=self.global_versioned, **self.class_assignment_categories[key]))

        return cat_list

    def calculate_percentiles(self) -> None:

        # df['Percentile'] = df['Score'].apply(lambda x: percentileofscore(df['Score'], x, kind='rank'))

        #scores = self.gbook['Course Total']
        
        # 'rank' method gives the percentage of scores less than or equal to the score 
        # so the highest score gets 100 percentile, lowest score is 1/N percentile
        self.gbook['Percentile'] = \
            self.gbook['Course Total'].apply(lambda x: int(percentileofscore(self.gbook['Course Total'], x, kind='rank')))

        # for 100 percentile, add to the Notes for this student 
        # NOTE: wont work if multiple students have the same top score... oh well for now (Github AI added the 'oh well for now' and that's hilarious)
        for index, row in self.gbook.iterrows():
            if row['Percentile'] >= 100.0:
                self.gbook.at[index, 'Notes'] += ' Highest grade in class! '


    def calculate_grades(self, letter_grades: dict[str,float]) -> None:

        # Create or clear out category total columns 
        self._create_assignment_categories_total_columns()

        # Add or clear out additional columns 
        self.gbook['Course Total'] = 0.0
        self.gbook['Percentile'] = 0.0
        self.gbook['Letter Grade'] = 'F'
        self.gbook['Grade Boundary'] = False
        self.gbook['Notes'] = ''

        self.course_columns = []
        self.course_columns.append('Percentile')
        self.course_columns.append('Course Total')
        #self.course.columns.append('Rank')
        self.course_columns.append('Letter Grade')
        self.course_columns.append('Grade Boundary')
        self.course_columns.append('Notes')


        # Given we might drop assignments and such, we will iterate over each student (row) and calculate their total points
        # for each category. 
        for index, row in self.gbook.iterrows():
            print(f"Working on student {row['Last']}")
            total_points = 0.0

            # If the student has a note, add it to the notes column
            for note in self.class_metadata['notes']:
                if note[0] == row['Last']:
                    self.gbook.at[index, 'Notes'] += note[1] + ' '


            # Calculate the total points for each category
            for cat in self.assignment_categories:
                #print(f'\tCategory: {cat.name}, {row[cat.items].tolist()}')

                # If the student has an overwrite, do it here
                for overwrite in self.class_metadata['overwrites']:
                    if overwrite[0] == row['Last'] and overwrite[1] in cat.items:
                        print(f"\tOverwriting {overwrite[1]} with {overwrite[2]}")
                        self.gbook.at[index, overwrite[1]] = overwrite[2]
                        row.at[overwrite[1]] = overwrite[2]
                        self.gbook.at[index, 'Notes'] += f"Overwrote {overwrite[1]} with {overwrite[2]}. "

                # If we need to do score replacements, do them here
                for replacement in self.class_metadata['grade_replacements']:
                    if cat.name == replacement[1]:
                        self.check_grade_replacements(replacement,cat,index,row)

                num_prorates = 0 
                for prorate in self.class_metadata['prorates']:
                    if prorate[0] == row['Last'] and prorate[1] == cat.name:
                        num_prorates = prorate[2]
                        print(f"\tProrating {cat.name} for {row['Last']} by {num_prorates} items.")
                        self.gbook.at[index, 'Notes'] += f"Prorated {cat.name} by {num_prorates} items. "

                cat_total = cat.calculate_category_total(row[cat.items].tolist(), num_prorates )
                total_points += cat_total*cat.course_points
                self.gbook.at[index, cat.name + ' Total'] = cat_total

                # If the student has an category total overwrite, do it here
                for overwrite in self.class_metadata['overwrites']:
                    if overwrite[0] == row['Last'] and overwrite[1] == cat.name + ' Total':
                        print(f"\tOverwriting {cat.name} Total with {overwrite[2]}")
                        self.gbook.at[index, cat.name + ' Total'] = overwrite[2]
                        row.at[cat.name + ' Total'] = overwrite[2]
                        self.gbook.at[index, 'Notes'] += f"Overwrote {cat.name} Total with {overwrite[2]}. "

            # Set the course total
            self.gbook.at[index, 'Course Total'] = total_points


            # If we overwrite the course total, do it here
            for overwrite in self.class_metadata['overwrites']:
                if overwrite[0] == row['Last'] and overwrite[1] == 'Course Total':
                    print(f"\tOverwriting Course Total with {overwrite[2]}")
                    old_score = self.gbook.at[index, 'Course Total']
                    self.gbook.at[index, 'Course Total'] = overwrite[2]
                    total_points = overwrite[2]  # Update total_points to the new value
                    row.at['Course Total'] = overwrite[2]
                    self.gbook.at[index, 'Notes'] += f"Overwrote Course Total from {old_score} with {overwrite[2]}. "

            # Determine the letter grade based on the total points
            for letter, boundary in letter_grades.items():
                if total_points >= boundary:
                    self.gbook.at[index, 'Letter Grade'] = letter
                    break

            # If we overwrite the letter grade, do it here
            student_done = False 
            for overwrite in self.class_metadata['overwrites']:
                if overwrite[0] == row['Last'] and overwrite[1] == 'Letter Grade':
                    print(f"\tOverwriting Letter Grade with {overwrite[2]}")
                    old_letter = self.gbook.at[index, 'Letter Grade']
                    self.gbook.at[index, 'Letter Grade'] = overwrite[2]
                    letter = overwrite[2]  # Update letter to the new value
                    row.at['Letter Grade'] = overwrite[2]
                    self.gbook.at[index, 'Notes'] += f"Overwrote Letter Grade from {old_letter} with {overwrite[2]}. "
                    # Continue to the next student since we already set the letter grade
                    student_done = True

            if student_done:
                continue 

            # Tests if they are on the a boundary (skipped if we overwrote the letter grade)
            for letter2, boundary in letter_grades.items():
                if letter2 == letter:
                    break
                if total_points + self.class_metadata['boundary_points'] >= boundary and letter2 != letter:
                    self.gbook.at[index, 'Grade Boundary'] = True
                    print(f'\tStudent {row["Last"]} is on the boundary for {letter2}.')
                    #self.gbook.at[index, 'Notes'] += f"On the boundary for {letter2}. "
                    break 

        # Calculate percentiles after all grades are assigned
        self.calculate_percentiles()
        print("Grades calculated and percentiles assigned.")


    def check_grade_replacements(self, replacement: list[str], cat_class, index, row) -> None:
        """
        Check if the grade replacement is valid and perform the replacement.
        :param replacement: list[str], Grade replacement information.
        """


        item, category, mode = replacement

        if mode not in ['one', 'all']:
            raise ValueError("Mode must be either 'one' or 'all'")

        # will just replace raw number for now... I gotta get this done 
        item_score = row[item] 

        # get all the scores for the category
        cat_scores = row[cat_class.items].tolist()

        cat_scores_arr = np.array(cat_scores, dtype=float)
        cat_scores_arr = np.nan_to_num(cat_scores, nan=0.0)
        sorted_indices = np.argsort(cat_scores_arr)
        cat_scores_arr = cat_scores_arr[sorted_indices]

        if item_score > cat_scores_arr[0]:
            replace_item_idx = sorted_indices[0]
            replace_item = cat_class.items[replace_item_idx]
            print(f"Replacing {replace_item} score {cat_scores_arr[0]} with {item_score}")
            self.gbook.at[index, replace_item] = item_score
            row.at[replace_item] = item_score
            self.gbook.at[index, 'Notes'] += f"Replaced {replace_item} score {cat_scores_arr[0]} with {item} score {item_score}. "
        return


        '''
        for replace_item in cat_class.items:
            cur_score = row[replace_item]
            if(item_score > cur_score):
                print(f"Replacing {replace_item} score {cur_score} with {item_score}")
                self.gbook.at[index, replace_item] = item_score
                row.at[replace_item] = item_score
                self.gbook.at[index, 'Notes'] += f"Replaced {replace_item} score {cur_score} with {item} score {item_score}. "
                return 
        '''
