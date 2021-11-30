#pragma once

#include "Common.h"

using Aga::Controls::Tree::Node;

namespace DPlus {

	public ref class fittingEntity : Node {
	public:
		Aga::Controls::Tree::TreeModel^ tree;
		System::String ^name;
		bool selected;
		bool checked;

		void ClearChecked() {
			for each (Node^ nd in Nodes)
			{
				fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nd);
				if(fnd) {
					fnd->ClearChecked();
				}
			}
			IsChecked = false;
		}

		void CheckParents() {
			fittingEntity^ fParent = dynamic_cast<fittingEntity^>(Parent);
			if(fParent) {
				fParent->IsChecked = true;
				fParent->CheckParents();
			}
		}

		virtual property System::String ^Text {
			System::String ^get() override {
				return name;
			}
		}

		virtual property bool Checked {
			bool get() {
				return IsChecked;
			}
			void set(bool value) {
				if(!value)
					return;
				// Change this to a radio button
				for each (Node^ nd in tree->Nodes)
				{
					fittingEntity^ fnd = dynamic_cast<fittingEntity^>(nd);
					if(fnd) {
						fnd->ClearChecked();
					}
				}
				IsChecked = value;
				CheckParents();
			}
		}

		virtual System::String ^ToString() override {
			return name;
		}

		fittingEntity(System::String ^Name, Aga::Controls::Tree::TreeModel^ Tree)
			: name(Name), tree(Tree)
		{
			selected = false;
			IsChecked = false;
			Node::Tag  = Name;
		}

		~fittingEntity() {}


	};

} // namespace DPlus