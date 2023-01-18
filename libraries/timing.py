"""
This file contains a class used to check the computation time of a calculation.
"""
import time
from math import floor

class CheckTime:
    """
    A class used to check computation time.
    
    Parameters
    ----------
    None
    """
    def __init__(self):
        self.start = time.time()
        
    def time_convert(
        self, 
        time: float
    ) -> tuple[float, float, float]:
        """
        Used to convert a time to hours, minutes, and seconds.
        
        Parameters
        ----------
        time : float
            The total time elapsed.

        Returns
        -------
        time_hrs : float
            Hours
        time_min : float
            Minutes
        time_sec : float
            Seconds
        """
        time_hrs = 0
        time_min = 0
        time_sec = 0

        if time >= 3600:
            time_hrs = time / 3600
            time_min = (time_hrs - floor(time_hrs)) * 60
            time_sec = round((time_min - floor(time_min)) * 60, 2)

            time_hrs = floor(time_hrs)
            time_min = floor(time_min)

        elif (time <= 3600) and (time >= 60):
            time_min = time / 60
            time_sec = round((time_min - floor(time_min)) * 60, 2) 

            time_min = floor(time_min)

        else:
            time_sec = round(time, 2)
            
        return time_hrs, time_min, time_sec

    def end_time(self) -> None:
        """
        Used to compute the total time elapsed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        start = self.start
        end = time.time()
        total_time = end - start
        hrs, mins, secs = self.time_convert(total_time)

        print("Time for training completion:", \
              hrs, "hrs", \
              mins, "min", \
              secs, "sec")
        print("\n")

        return None




