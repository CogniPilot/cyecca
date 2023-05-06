class Array(ABC):
    
    @abstractmethod
    def __init__(self, data):
        ...

    @abstractmethod
    def __getitem__(self, indices):
        ...

    @abstractmethod
    def __setitem__(self, indices, value):
        ...
    
    @abstractmethod
    def __mul__(self, right):
        ...
    
    @abstractmethod
    def __rmul__(self, left):
        ...
    
    @abstractmethod
    def __add__(self, right):
        ...

    @abstractmethod
    def __sub__(self, right):
        ...

    @abstractmethod
    def __repr__(self):
        ...


class ArrayNumpy(Array):
    
    def __init__(self, data):
        self.data = np.array(data)

    def __getitem__(self, indices):
        return self.data[indices]

    def __setitem__(self, indices, val):
        self.data[indices] = val

    def __mul__(self, right):
        return ArrayNumpy(self.data@right.data)
    
    def __rmul__(self, left):
        return ArrayNumpy(left*self.data)
    
    def __add__(self, right):
        return ArrayNumpy(self.data + right.data)

    def __sub__(self, right):
        return ArrayNumpy(self.data - right.data)

    def __repr__(self):
        return repr(self.data)

a = ArrayNumpy([[1, 2]])
a[0, 0] = 3
a - 2*a