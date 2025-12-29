'''
CACAO Assignment Category Class Definition
'''

import pandas as pd 
from typing import Any
import numpy as np


class AssignmentCategory:

    def __init__(self, gbook: pd.DataFrame, global_versioned: dict[str, Any], name: str, 
        items: list[str], point_totals: list[float] | None = None, 
        combine_items: list[tuple[str,str]] | None = None, extra_credit_items: list[str] | None = None, 
        course_points: int = 100, equal_weight: bool = False, weights: list[float] | None = None, 
        drop_lowest: int = 0, is_overperfect_extracredit: bool = False, version: str = 'v1') -> None :
        """
        Initialize an AssignmentCategory object.
        :param name: str, Name of the assignment category.
        :param gbook: pd.DataFrame, unaltered gradebook DataFrame. (not stored)
        :param global_versioned: dict[str, Any], Global versioned settings for the program.
        :param items: list[str], List of item names in the category.
        :param point_totals: list[float] or None, List of point totals for each item. If None, will be calculated from gbook.
        :param combine_items: list[tuple[str,str]] or None, List of tuples where each tuple contains two item names to be combined.
            These items must be in the items list 
        :param extra_credit_items: list[str] or None, List of item names that are extra credit.
            These items must be in the items list
        :param course_points: int = 100, Total points for the course.
        :param equal_weight: bool = False, If True, all items in the category are weighted equally.
        :param weights: list[float] or None, List of weights for each item. If equal_weight is True, this will be ignored.
        :param drop_lowest: int = 0 , Number of lowest scoring items to drop from the category.
        :param is_overperfect_extracredit: bool = False, If True, if points exceed the point total = extra credit, else ignored.
        :param version: str = 'v1', Version of the gradebook format. Used for compatibility with different versions.
        """

        # Initialize name
        self.name = name
        assert isinstance(self.name,str), "Name must be a string"

        # Version 
        self.global_versioned = global_versioned
        assert isinstance(self.global_versioned, dict), "Global versioned must be a dictionary"

        self.version = version
        assert isinstance(self.version, str), "Version must be a string"

        # Course points 
        self.course_points = course_points
        assert self.course_points > 0, "Course points must be greater than 0"

        # Column names in the gradebook and their point totals
        self.items = items
        assert self._in_gradebook(self.items, gbook), f"All items must be in the gradebook {self.items=}"
        assert len(self.items) > 0, "There must be at least one item in the category"

        #self.point_totals = point_totals if point_totals is not None else self._calc_point_totals(gbook, items)
        self.point_totals = self._calc_point_totals(gbook, items)
        assert len(self.point_totals) == len(self.items), "Points must match the number of items"


        # ignoring for now 
        # Possible combine and extra credit items, checks if everything is in the gradebook 
        self.combine_items = combine_items if combine_items is not None else [] # ignoring for now 
        #assert self._in_gradebook(self.combine_items, gbook), "All combine items must be in the gradebook" # ignoring for now 
        
        # Extra credit items
        self.extra_credit_items = extra_credit_items if extra_credit_items is not None else [] 
        assert self._in_gradebook(self.extra_credit_items, gbook), "All extra credit items must be in the gradebook"

        # Drop lowest items
        self.drop_lowest = drop_lowest
        assert self.drop_lowest >= 0 and isinstance(self.drop_lowest, int), "Drop lowest must be a non-negative integer"
        assert self.drop_lowest < len(self.items), "Drop lowest must be less than the number of items in the category"

        # Weightings 
        self.equal_weight = equal_weight
        if equal_weight:
            self.weights = [1.0] * len(items)
        else:
            self.weights = weights if weights is not None else self.point_totals
        assert len(self.weights) == len(self.items),"Weights must match the number of items"
        assert all(w >= 0 for w in self.weights), "Weights must be non-negative"

        # Check if all weights are identical
        if all(w == self.weights[0] for w in self.weights):
            self.equal_weight = True
            self.weights = [1.0] * len(self.items)

        # Is this category overperfect extra credit?
        self.is_overperfect_extracredit = is_overperfect_extracredit


    def calculate_category_total(self, scores: list[float], num_prorates = 0) -> float:

        # These are copies by construction 
        scores_arr = np.array(scores, dtype=float)
        scores_arr = np.nan_to_num(scores, nan=0.0)
        points_arr = np.array(self.point_totals, dtype=float)
        points_arr = np.nan_to_num(points_arr, nan=0.0)
        weights_arr = np.array(self.weights, dtype=float)
        weights_arr = np.nan_to_num(weights_arr, nan=0.0)

        # spot check 
        assert len(scores_arr) == len(points_arr) == len(weights_arr), "Scores, points, and weights must have the same length"
        assert all(points_arr > 0), "All point totals must be greater than 0"

        # Sort arrays based on scores_arr (ascending order)
        sorted_indices = np.argsort(scores_arr)
        scores_arr = scores_arr[sorted_indices]
        points_arr = points_arr[sorted_indices]
        weights_arr = weights_arr[sorted_indices]
        # Derived fractional scores
        frac_scores_arr = scores_arr / points_arr

        drop_me = int(max(0, self.drop_lowest, num_prorates))
        assert drop_me < len(scores_arr), "Drop lowest must be less than the number of items in the category (likely prorating issue)"

        # If equally weighted, zero out the first N weights 
        if self.equal_weight:
            if drop_me > 0:
                weights_arr[:drop_me] = 0.0

            cat_total = np.sum(weights_arr * frac_scores_arr) / np.sum(weights_arr)

        else:
            # For now will asume weights = points and subtract N of the the average
            # point total from the total possible points. 
            mean_points = np.mean(points_arr)

            cat_total = np.sum(scores_arr) / (  np.sum(points_arr) - (drop_me * mean_points) )

        # If this is overperfect extra credit, we will cap the total at 1.0 TODO (fix this)
        if not self.is_overperfect_extracredit:
            cat_total = min(cat_total, 1.0)
        # If the total is less than 0, set it to 0
        if cat_total < 0:
            cat_total = 0.0

        print(round(cat_total,3),end=' ')

        return(cat_total)





    def _in_gradebook(self, items: list[str], gbook: pd.DataFrame) -> bool:
        """
        Check if all items are in the gradebook.
        :param items: list[str], List of item names to check.
        :param gbook: pd.DataFrame, Gradebook DataFrame.
        :return: bool, True if all items are in the gradebook, False otherwise.
        """
        
        if len(items) == 0:
            return True 

        cols = gbook.columns.tolist()
        for item in items:
            if isinstance(item, tuple):
                # If item is a tuple, check both items in the tuple
                for sub_item in item:
                    if sub_item not in cols:
                        print(f"Item '{sub_item}' not found in gradebook columns: {cols}")
                        return False
            elif isinstance(item, str):
                # If item is a string, check the string
                if item not in cols:
                    print(f"Item '{item}' not found in gradebook columns: {cols}")
                    return False
            else:
                # Should never happen, but just in case
                print(f"Item '{item}' is not a string or tuple: {type(item)}")
                return False
        return True

    def get_course_points(self) -> int:
        """
        Get the total course points for the category.
        :return: int, Total course points for the category.
        """
        return self.course_points

    def get_point_totals(self) -> list[float]:
        """
        Get the point totals for each item in the category.
        :return: list[float], List of point totals for each item.
        """
        return self.point_totals

    def get_column_names(self) -> list[str]:
        """
        Get the column names for the items in the category.
        :return: list[str], List of item names in the category.
        """
        return set( self.items )

    def _calc_weights(self,gbook: pd.DataFrame, items: list[str]) -> pd.DataFrame:
        """
        Calculate the weights for each item in the category based on the point totals.
        If the item is not found, it will break.
        :param gbook: pd.DataFrame, Gradebook DataFrame.
        :param items: list[str], List of item names in the category.
        :return: pd.DataFrame, DataFrame with weights for each item.
        """

        # Get the point totals for each item
        point_totals = self._calc_point_totals(gbook, items)

        # Calculate weights
        weights = point_totals / point_totals.sum()

        return weights


    def _calc_point_totals(self, gbook: pd.DataFrame, items: list[str]) -> pd.DataFrame:
        """
        Get the point totals for each item in the category from the gradebook.
        If the item is not found, it will break.
        """

        # Find the row where the first column has '    Points Possible'
        points_row = gbook[gbook.iloc[:, 0] == self.global_versioned["points_possible"][self.version]]

        point_totals = [float(points_row[item].iloc[0]) for item in items]

        return point_totals



