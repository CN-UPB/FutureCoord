from collections.abc import MutableMapping

DEFAULT = object()


class BiDict(MutableMapping):
    def __init__(self, mirror=None, val_btype=set, key_map=None):
        self._dict = {}
        self.mirror = mirror

        # datastructure used to store values of dictionary; set by default
        self.val_btype = val_btype

        # key map is the identity function by default
        self.key_map = lambda key: key if key_map is None else key_map(key)

        if mirror is not None:
            mirror.mirror = self

    def __contains__(self, key):
        return key in self._dict

    def __setitem__(self, key, value, inv=False):
        if self.val_btype is set:
            self._dict.setdefault(self.key_map(key), set()).add(value)
        elif self.val_btype is list:
            self._dict.setdefault(self.key_map(key), list()).append(value)
        else:
            raise

        if not inv:
            self.mirror.__setitem__(value, key, inv=True)

    def __getitem__(self, key):
        key = self.key_map(key)
        if key in self._dict:
            return self._dict[key]

        # handle default behavior for unknown keys
        return self.val_btype()

    def __delitem__(self, key):
        # delete (key, value) pairs from inverse dictionary
        vals = set(self._dict[self.key_map(key)])
        for val in vals:
            self.mirror[val].remove(key)

            # remove keys from inverse dictionary if their list is empty
            if not self.mirror[val]:
                del self.mirror[val]

        # delete (key, value) pairs from dictionary
        del self._dict[self.key_map(key)]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def _keytransform(self, key):
        return key

    def __repr__(self):
        return repr(self._dict)

    def pop(self, key, default=DEFAULT):
        if key in self or default is DEFAULT:
            key = self.key_map(key)
            value = self[key]
            del self[key]
            return value

        else:
            return default
